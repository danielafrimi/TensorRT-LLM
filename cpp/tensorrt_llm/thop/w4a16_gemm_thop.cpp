#include "w4a16_gemm_thop.h"

// TensorRT-LLM CUTLASS kernel headers
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/weight_only_quant_op.h" // Adjusted path
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <ATen/cuda/CUDAContext.h> // For at::cuda::getCurrentCUDAStream
#include <algorithm>               // For std::min
#include <cuda_runtime.h>          // For cudaError_t, cudaDeviceSynchronize, cudaGetErrorString
#include <iomanip>                 // For std::fixed, std::setprecision
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept> // For std::runtime_error
#include <string>
#include <tuple>
#include <vector>

// Helper to get/manage the CUTLASS runner instance and tactic map (simplified)
namespace
{

// Define the specific runner type you want to use.
// This assumes FP16 activation
// group-wise scale-only quantization, and FP16 for scales, bias, output.
using ActivationT = half;
using WeightPackedT = uint8_t; // INT4 weights (packed in uint8_t)
using ScaleT = half;
using BiasT = half;
using OutputT = half;

const cutlass::WeightOnlyQuantOp QuantMode = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;

using W4A16Runner = tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<ActivationT, WeightPackedT,
    QuantMode, ScaleT, BiasT, OutputT>;

W4A16Runner& get_w4a16_runner_instance()
{ // Renamed to avoid potential clashes
    static W4A16Runner runner_instance;
    return runner_instance;
}

// Placeholder for tactic map. Production code would populate this via profiling.
std::map<std::tuple<int, int, int, int>, tensorrt_llm::cutlass_extensions::CutlassGemmConfig> g_w4a16_tactics_map;
std::mutex g_w4a16_tactics_mutex;

} // namespace

torch::Tensor w4a16_gemm_op_forward(torch::Tensor A_tensor, torch::Tensor B_packed_tensor, torch::Tensor scales_tensor,
    torch::Tensor zeros_tensor, // Argument still received but will be ignored internally for this QuantMode
    torch::Tensor bias_tensor, int64_t group_size_long)
{

    // --- Input Checks (Essential!) ---
    TORCH_CHECK(A_tensor.is_cuda(), "Activation tensor A must be on CUDA");
    TORCH_CHECK(B_packed_tensor.is_cuda(), "Packed Weight tensor B must be on CUDA");
    TORCH_CHECK(scales_tensor.is_cuda(), "Scales tensor must be on CUDA");

    TORCH_CHECK(A_tensor.scalar_type() == torch::kFloat16, "Activation A must be FP16");
    TORCH_CHECK(B_packed_tensor.scalar_type() == torch::kInt8, "Packed Weights B must be Int8 (representing Int4s)");
    TORCH_CHECK(scales_tensor.scalar_type() == torch::kFloat16, "Scales must be FP16");

    TORCH_CHECK(A_tensor.is_contiguous(), "Activation tensor A must be contiguous");
    TORCH_CHECK(B_packed_tensor.is_contiguous(), "Packed weight tensor B must be contiguous");
    TORCH_CHECK(scales_tensor.is_contiguous(), "Scales tensor must be contiguous");

    // --- Get Dimensions ---
    TORCH_CHECK(A_tensor.dim() >= 2, "Activation A must be at least 2D");
    // Assuming B_packed_tensor is [K, N_packed] where N_packed = N_orig // 2
    TORCH_CHECK(B_packed_tensor.dim() == 2, "Packed Weights B must be 2D");

    int M = 0;
    int K_act = 0;
    if (A_tensor.dim() == 2)
    {
        M = A_tensor.size(0);
        K_act = A_tensor.size(1);
    }
    else
    {                              // Handle batched input (e.g. >2D tensor)
        M = A_tensor.size(0);      // First dim is batch
        for (int i = 1; i < A_tensor.dim() - 1; ++i)
            M *= A_tensor.size(i); // Flatten batch dimensions
        K_act = A_tensor.size(A_tensor.dim() - 1);
    }
    // add print here of M, K_act, N_orig
    std::cout << "M: " << M << ", K_act: " << K_act << std::endl;

    int K_weights = B_packed_tensor.size(0);
    int N_packed_int4 = B_packed_tensor.size(1);
    int N_orig = N_packed_int4 * 2;

    TORCH_CHECK(
        K_act == K_weights, "K dimension of A (", K_act, ") must match K dimension of B_packed (", K_weights, ")");
    int K = K_act;
    int group_size = static_cast<int>(group_size_long);

    // --- Prepare Output Tensor ---
    std::vector<int64_t> output_shape;
    if (A_tensor.dim() == 2)
    {
        output_shape.clear(); // Clear before assigning new shape
        output_shape.push_back(static_cast<int64_t>(M));
        output_shape.push_back(static_cast<int64_t>(N_orig));
    }
    else
    {
        output_shape.clear(); // Clear before assigning new shape
        for (int i = 0; i < A_tensor.dim() - 1; ++i)
            output_shape.push_back(A_tensor.size(i));
        output_shape.push_back(N_orig);
    }
    torch::Tensor C_tensor = torch::empty(output_shape, A_tensor.options().dtype(torch::kFloat16));

    // --- Get Data Pointers ---
    void* A_ptr = A_tensor.data_ptr();
    void* B_packed_ptr = B_packed_tensor.data_ptr();
    void* scales_ptr = scales_tensor.data_ptr();

    // No need to get zeros_ptr since QuantMode is FINEGRAINED_SCALE_ONLY
    // We will pass nullptr to the runner.gemm call for the zeros argument.
    // We can still perform checks on the input zeros_tensor if it's provided,
    // just don't use its pointer for this QuantMode.
    if (zeros_tensor.defined() && zeros_tensor.numel() > 0)
    { // todo delete it
        TORCH_CHECK(zeros_tensor.is_cuda(), "Zeros tensor must be on CUDA if provided and not empty");
        TORCH_CHECK(
            zeros_tensor.scalar_type() == torch::kFloat16, "Zeros tensor must be FP16 if provided and not empty");
        TORCH_CHECK(zeros_tensor.is_contiguous(), "Zeros tensor must be contiguous if provided and not empty");
        // We don't assign zeros_tensor.data_ptr() to zeros_ptr here.
    }
    // The check for FINEGRAINED_SCALE_AND_ZEROS needing a tensor is removed/no longer relevant here
    // because we are hardcoded to FINEGRAINED_SCALE_ONLY.
    void* zeros_ptr = nullptr; // Explicitly ensure zeros_ptr is nullptr for the call

    void* bias_ptr = nullptr;
    if (bias_tensor.defined() && bias_tensor.numel() > 0)
    {
        TORCH_CHECK(bias_tensor.is_cuda(), "Bias tensor must be on CUDA if provided and not empty");
        TORCH_CHECK(bias_tensor.scalar_type() == torch::kFloat16, "Bias tensor must be FP16 if provided and not empty");
        TORCH_CHECK(bias_tensor.is_contiguous(), "Bias tensor must be contiguous");
        bias_ptr = bias_tensor.data_ptr();
    }
    void* C_ptr = C_tensor.data_ptr();

    // --- Get CUTLASS Runner and Config ---
    W4A16Runner& runner = get_w4a16_runner_instance();
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig gemm_config_to_run;
    std::tuple<int, int, int, int> tactic_key = std::make_tuple(M, N_orig, K, group_size);
    {
        std::lock_guard<std::mutex> lock(g_w4a16_tactics_mutex);
        auto it = g_w4a16_tactics_map.find(tactic_key);
        if (it != g_w4a16_tactics_map.end())
        {
            gemm_config_to_run = it->second;
        }
        else
        {
            int sm_version = tensorrt_llm::common::getSMVersion();
            std::cout << "[WARNING] w4a16_gemm_thop: Using first *valid* available CUTLASS config. PERFORMANCE MAY BE "
                         "SUBOPTIMAL."
                      << std::endl;
            auto all_configs = runner.getConfigs();
            TORCH_CHECK(!all_configs.empty(), "No CUTLASS GEMM configs available for W4A16 runner.");
            std::cout << "[DEBUG] Available configs from runner.getConfigs():" << std::endl;
            for (auto const& config : all_configs)
            {
                std::cout << "  - TileConfSM80: " << static_cast<int>(config.tile_config_sm80)
                          << ", Stages: " << config.stages
                          << ", SplitKStyle: " << static_cast<int>(config.split_k_style)
                          << ", SplitKFactor: " << config.split_k_factor
                          << ", CudaKernel: " << (config.enableCudaKernel ? "yes" : "no") << std::endl;
            }
            gemm_config_to_run = all_configs[0];
            if (sm_version >= 89 && gemm_config_to_run.stages == 2)
            {
                std::cout << "[WARNING] First config had stages=2 on SM89+, attempting to find another." << std::endl;
                bool found_valid_config = false;
                for (auto const& config : all_configs)
                {
                    bool config_ok = true;
                    if (sm_version >= 89 && config.stages == 2)
                    {
                        config_ok = false;
                    }
                    if (config_ok)
                    {
                        gemm_config_to_run = config;
                        found_valid_config = true;
                        std::cout << "Selected valid fallback config: Stages=" << config.stages << std::endl;
                        break;
                    }
                }
                TORCH_CHECK(found_valid_config, "Could not find a valid CUTLASS config (stages!=2) for SM 89+.");
            }
            else
            {
                std::cout << "Selected fallback config: Stages=" << gemm_config_to_run.stages << std::endl;
            }
            g_w4a16_tactics_map[tactic_key] = gemm_config_to_run;
        }
    }

    // --- Workspace ---
    size_t workspace_bytes = runner.getWorkspaceSize(M, N_orig, K);
    torch::Tensor workspace_tensor = torch::empty(
        {static_cast<int64_t>(workspace_bytes)}, torch::TensorOptions().dtype(torch::kUInt8).device(A_tensor.device()));
    char* workspace_ptr = workspace_bytes > 0 ? reinterpret_cast<char*>(workspace_tensor.data_ptr()) : nullptr;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A_tensor.device().index());

    // Synchronize device before attempting to read tensor data on CPU for printing.
    // This helps ensure that any preceding CUDA operations (like runner setup, workspace allocation)
    // are complete and memory is in a consistent state, which might be necessary if
    // A_ptr becomes inaccessible after those operations due to async execution or context issues.
    std::cout << "[DEBUG] w4a16_gemm_thop: Attempting cudaDeviceSynchronize() before printing tensor contents."
              << std::endl;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "[ERROR] cudaDeviceSynchronize failed in w4a16_gemm_thop: " << cudaGetErrorString(err) << " ("
                  << __FILE__ << ":" << __LINE__ << ")" << std::endl;
        throw std::runtime_error(
            std::string("[CUDA Error] w4a16_gemm_thop: cudaDeviceSynchronize failed: ") + cudaGetErrorString(err));
    }
    std::cout << "[DEBUG] w4a16_gemm_thop: cudaDeviceSynchronize() completed." << std::endl;

    // --- Print Tensor Contents (Debug) ---
    std::cout << "[DEBUG] w4a16_gemm_thop - Tensor contents before runner.gemm call:" << std::endl;

    // --- Detailed A_tensor check before attempting to print its data ---
    std::cout << "[DEBUG] Pre-Print A_tensor metadata:" << std::endl;
    std::cout << "  A_tensor.defined(): " << A_tensor.defined() << std::endl;
    if (A_tensor.defined())
    {
        std::cout << "  A_tensor.is_cuda(): " << A_tensor.is_cuda() << std::endl;
        std::cout << "  A_tensor.is_contiguous(): " << A_tensor.is_contiguous() << std::endl;
        std::cout << "  A_tensor.numel(): " << A_tensor.numel() << std::endl;
        std::cout << "  A_tensor.dim(): " << A_tensor.dim() << std::endl;
        for (int d = 0; d < A_tensor.dim(); ++d)
        {
            std::cout << "    A_tensor.size(" << d << "): " << A_tensor.size(d) << std::endl;
        }
        std::cout << "  Calculated M (for print loop): " << M << std::endl;
        std::cout << "  Calculated K_act (for print loop): " << K_act << std::endl;
        std::cout << "  A_ptr (re-checked address): " << A_tensor.data_ptr() << std::endl; // Get current data_ptr
    }
    else
    {
        std::cout << "  A_tensor is NOT defined!" << std::endl;
    }
    // --- End Detailed A_tensor check ---

    // Print A_tensor (Activation)
    if (A_ptr && A_tensor.defined() && A_tensor.numel() > 0)
    { // Added A_tensor.defined() and numel check
        std::cout << "  A_tensor (first 2 rows, up to 5 elements each, copied to CPU for printing):" << std::endl;

        int rows_to_print_a = std::min(2, M);
        int cols_to_print_a = std::min(5, K_act);
        size_t elements_to_copy_a = static_cast<size_t>(rows_to_print_a) * K_act; // Copy full relevant rows
        size_t bytes_to_copy_a = elements_to_copy_a * sizeof(ActivationT);

        if (elements_to_copy_a > 0 && bytes_to_copy_a <= A_tensor.numel() * sizeof(ActivationT))
        {
            std::vector<ActivationT> a_cpu_buffer(elements_to_copy_a);
            std::cout << "[DEBUG] Attempting cudaMemcpy for A_tensor: " << bytes_to_copy_a << " bytes." << std::endl;
            cudaError_t memcpy_err_a = cudaMemcpy(a_cpu_buffer.data(), A_ptr, bytes_to_copy_a, cudaMemcpyDeviceToHost);
            if (memcpy_err_a != cudaSuccess)
            {
                std::cerr << "[ERROR] cudaMemcpy DtoH for A_tensor failed: " << cudaGetErrorString(memcpy_err_a)
                          << std::endl;
            }
            else
            {
                std::cout << "[DEBUG] cudaMemcpy for A_tensor successful." << std::endl;
                for (int i = 0; i < rows_to_print_a; ++i)
                {
                    std::cout << "    Row " << i << ": ";
                    for (int j = 0; j < cols_to_print_a; ++j)
                    {
                        // Access data from the CPU buffer
                        std::cout << static_cast<float>(a_cpu_buffer[i * K_act + j]) << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
        else
        {
            std::cout << "[DEBUG] Skipping A_tensor print due to zero elements to copy or invalid copy size."
                      << std::endl;
            std::cout << "  elements_to_copy_a: " << elements_to_copy_a << ", bytes_to_copy_a: " << bytes_to_copy_a
                      << std::endl;
            std::cout << "  A_tensor.numel() * sizeof(ActivationT): " << A_tensor.numel() * sizeof(ActivationT)
                      << std::endl;
        }
    }

    // Print B_packed_tensor (Weights)
    if (B_packed_ptr && B_packed_tensor.defined() && B_packed_tensor.numel() > 0)
    {
        // WeightPackedT* B_packed_data = static_cast<WeightPackedT*>(B_packed_ptr);
        std::cout << "  B_packed_tensor (first 2 rows, up to 5 elements each, copied to CPU for printing):"
                  << std::endl;

        int rows_to_print_b = std::min(2, K_weights);
        int cols_to_print_b = std::min(5, N_packed_int4);
        size_t elements_to_copy_b = static_cast<size_t>(rows_to_print_b) * N_packed_int4; // Copy full relevant rows
        size_t bytes_to_copy_b = elements_to_copy_b * sizeof(WeightPackedT);

        if (elements_to_copy_b > 0 && bytes_to_copy_b <= B_packed_tensor.numel() * sizeof(WeightPackedT))
        {
            std::vector<WeightPackedT> b_cpu_buffer(elements_to_copy_b);
            std::cout << "[DEBUG] Attempting cudaMemcpy for B_packed_tensor: " << bytes_to_copy_b << " bytes."
                      << std::endl;
            cudaError_t memcpy_err_b
                = cudaMemcpy(b_cpu_buffer.data(), B_packed_ptr, bytes_to_copy_b, cudaMemcpyDeviceToHost);

            if (memcpy_err_b != cudaSuccess)
            {
                std::cerr << "[ERROR] cudaMemcpy DtoH for B_packed_tensor failed: " << cudaGetErrorString(memcpy_err_b)
                          << std::endl;
            }
            else
            {
                std::cout << "[DEBUG] cudaMemcpy for B_packed_tensor successful." << std::endl;
                // B_packed_tensor shape: [K_weights, N_packed_int4]
                for (int i = 0; i < rows_to_print_b; ++i)
                {
                    std::cout << "    Row " << i << ": ";
                    for (int j = 0; j < cols_to_print_b; ++j)
                    {
                        std::cout << static_cast<int>(b_cpu_buffer[i * N_packed_int4 + j]) << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
        else
        {
            std::cout << "[DEBUG] Skipping B_packed_tensor print due to zero elements to copy or invalid copy size."
                      << std::endl;
            std::cout << "  elements_to_copy_b: " << elements_to_copy_b << ", bytes_to_copy_b: " << bytes_to_copy_b
                      << std::endl;
            std::cout << "  B_packed_tensor.numel() * sizeof(WeightPackedT): "
                      << B_packed_tensor.numel() * sizeof(WeightPackedT) << std::endl;
        }
    }

    // Print scales_tensor
    if (scales_ptr && scales_tensor.defined() && scales_tensor.numel() > 0)
    {
        // ScaleT* scales_data = static_cast<ScaleT*>(scales_ptr);
        std::cout << "  scales_tensor (first 2 rows, up to 5 elements each, copied to CPU for printing):" << std::endl;
        // scales_tensor shape: [total_groups, N_orig]
        int total_groups_calc = (K + group_size - 1) / group_size; // Calculate total_groups for loop bound

        int rows_to_print_s = std::min(2, total_groups_calc);
        int cols_to_print_s = std::min(5, N_orig);
        size_t elements_to_copy_s = static_cast<size_t>(rows_to_print_s) * N_orig; // Copy full relevant rows
        size_t bytes_to_copy_s = elements_to_copy_s * sizeof(ScaleT);

        if (elements_to_copy_s > 0 && bytes_to_copy_s <= scales_tensor.numel() * sizeof(ScaleT))
        {
            std::vector<ScaleT> s_cpu_buffer(elements_to_copy_s);
            std::cout << "[DEBUG] Attempting cudaMemcpy for scales_tensor: " << bytes_to_copy_s << " bytes."
                      << std::endl;
            cudaError_t memcpy_err_s
                = cudaMemcpy(s_cpu_buffer.data(), scales_ptr, bytes_to_copy_s, cudaMemcpyDeviceToHost);

            if (memcpy_err_s != cudaSuccess)
            {
                std::cerr << "[ERROR] cudaMemcpy DtoH for scales_tensor failed: " << cudaGetErrorString(memcpy_err_s)
                          << std::endl;
            }
            else
            {
                std::cout << "[DEBUG] cudaMemcpy for scales_tensor successful." << std::endl;
                for (int i = 0; i < rows_to_print_s; ++i)
                {
                    std::cout << "    Row " << i << ": ";
                    for (int j = 0; j < cols_to_print_s; ++j)
                    {
                        std::cout << static_cast<float>(s_cpu_buffer[i * N_orig + j]) << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
        else
        {
            std::cout << "[DEBUG] Skipping scales_tensor print due to zero elements to copy or invalid copy size."
                      << std::endl;
            std::cout << "  elements_to_copy_s: " << elements_to_copy_s << ", bytes_to_copy_s: " << bytes_to_copy_s
                      << std::endl;
            std::cout << "  scales_tensor.numel() * sizeof(ScaleT): " << scales_tensor.numel() * sizeof(ScaleT)
                      << std::endl;
        }
    }

    // Print bias_tensor
    if (bias_ptr && bias_tensor.defined() && bias_tensor.numel() > 0)
    { // bias_ptr is nullptr if bias_tensor is not defined or empty
        // BiasT* bias_data = static_cast<BiasT*>(bias_ptr);
        std::cout << "  bias_tensor (first 1 row, up to 5 elements, copied to CPU for printing):" << std::endl;

        int cols_to_print_bias = std::min(5, N_orig); // N_orig is the width of bias
        size_t elements_to_copy_bias = static_cast<size_t>(cols_to_print_bias);
        size_t bytes_to_copy_bias = elements_to_copy_bias * sizeof(BiasT);

        if (elements_to_copy_bias > 0 && bytes_to_copy_bias <= bias_tensor.numel() * sizeof(BiasT))
        {
            std::vector<BiasT> bias_cpu_buffer(elements_to_copy_bias);
            std::cout << "[DEBUG] Attempting cudaMemcpy for bias_tensor: " << bytes_to_copy_bias << " bytes."
                      << std::endl;
            cudaError_t memcpy_err_bias
                = cudaMemcpy(bias_cpu_buffer.data(), bias_ptr, bytes_to_copy_bias, cudaMemcpyDeviceToHost);

            if (memcpy_err_bias != cudaSuccess)
            {
                std::cerr << "[ERROR] cudaMemcpy DtoH for bias_tensor failed: " << cudaGetErrorString(memcpy_err_bias)
                          << std::endl;
            }
            else
            {
                std::cout << "[DEBUG] cudaMemcpy for bias_tensor successful." << std::endl;
                std::cout << "    Row 0: ";
                for (int j = 0; j < cols_to_print_bias; ++j)
                {
                    std::cout << static_cast<float>(bias_cpu_buffer[j]) << " ";
                }
                std::cout << std::endl;
            }
        }
        else
        {
            std::cout << "[DEBUG] Skipping bias_tensor print due to zero elements to copy or invalid copy size."
                      << std::endl;
            std::cout << "  elements_to_copy_bias: " << elements_to_copy_bias
                      << ", bytes_to_copy_bias: " << bytes_to_copy_bias << std::endl;
            std::cout << "  bias_tensor.numel() * sizeof(BiasT): " << bias_tensor.numel() * sizeof(BiasT) << std::endl;
        }
    }
    else
    {
        std::cout << "  bias_tensor: Not provided or empty." << std::endl;
    }

    // --- Execute GEMM ---
    // Pass nullptr explicitly for the zeros argument, matching the FINEGRAINED_SCALE_ONLY QuantMode
    runner.gemm(A_ptr, B_packed_ptr, scales_ptr,
        nullptr, // NOTE: zeros is not used for this QuantMode
        bias_ptr,
        1.0f,    // Alpha
        C_ptr, M, N_orig, K, group_size, gemm_config_to_run, workspace_ptr, workspace_bytes, stream);

    return C_tensor;
}

// --- Register the Operator with PyTorch ---
// The "trtllm" namespace is what you'll use in Python (torch.ops.trtllm.w4a16_gemm)
TORCH_LIBRARY(tensorrt_llm, m)
{
    m.def("w4a16_gemm(Tensor A, Tensor B_packed, Tensor scales, Tensor zeros, Tensor bias, int group_size) -> Tensor",
        &w4a16_gemm_op_forward);
    // Example for a profiling function (not fully implemented here, just a stub for registration)
    // m.def("w4a16_gemm_profile(int M, int N, int K, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(tensorrt_llm, CUDA, m)
{
    m.impl("w4a16_gemm", &w4a16_gemm_op_forward);
}
