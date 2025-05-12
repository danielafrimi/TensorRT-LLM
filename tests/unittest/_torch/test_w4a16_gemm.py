import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

from tests.unittest.trt.quantization import _utils


def create_idenetity_matrix_after_unpacking(k, n):

    target_identity_ref_q_weight = torch.eye(k,
                                             n,
                                             dtype=torch.int8,
                                             device="cpu")

    unprocessed_weight_cpu = torch.zeros(k,
                                         n // 2,
                                         dtype=torch.int8,
                                         device="cpu")

    # 3. Pack the values from target_identity_ref_q_weight into unprocessed_weight_cpu
    for i in range(k):
        for j_packed in range(n // 2):
            # Value that should appear at the even column index in ref_q_weight (e.g., ref_q_weight[i, 0], ref_q_weight[i, 2], ...)
            val_for_even_output_idx = target_identity_ref_q_weight[i,
                                                                   2 * j_packed]
            # Value that should appear at the odd column index in ref_q_weight (e.g., ref_q_weight[i, 1], ref_q_weight[i, 3], ...)
            val_for_odd_output_idx = target_identity_ref_q_weight[i,
                                                                  2 * j_packed +
                                                                  1]

            # The unpacker places the lower 4 bits of packed_value into the even output index
            # and the higher 4 bits into the odd output index.
            # So, val_for_even_output_idx must go into the lower nibble,
            # and val_for_odd_output_idx must go into the higher nibble.
            packed_value = (val_for_odd_output_idx.item() << 4) | (
                val_for_even_output_idx.item() & 0x0F)
            unprocessed_weight_cpu[i, j_packed] = packed_value

    return unprocessed_weight_cpu


def run_w4a16_gemm_test(m, n, k, group_size, activation_dtype,
                        quantized_weight_dtype, has_zero, has_bias,
                        has_pre_quant, device):
    total_groups = (k + group_size - 1) // group_size
    print(f"total_groups: {total_groups}")

    activation = torch.randn(m, k, dtype=activation_dtype, device=device)
    scale = torch.ones(total_groups, n, dtype=activation_dtype,
                       device="cuda")  # todo change it back to random

    pre_quant_scale = torch.rand(1, k, dtype=activation_dtype, device="cuda")

    bias = torch.zeros(
        1, n, dtype=activation_dtype,
        device="cuda") if has_bias else None  # todo change it back to random

    zero = torch.randn(total_groups, n, dtype=activation_dtype,
                       device="cuda") if has_zero else None

    unprocessed_weight_cpu = create_idenetity_matrix_after_unpacking(k, n)

    # unprocessed_weight needs to be on the specified device for the GEMM operation
    unprocessed_weight = unprocessed_weight_cpu.to(device)

    # unprocessed_weight = torch.randint(
    #     -128,
    #     127,(k, n // 2),
    #     dtype=torch.int8,  # todo does it need to be uint or int?
    #     device="cuda")

    unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
    # Weights must be a CPU Tensor
    ref_q_weight = unpacker(
        unprocessed_weight.cpu())  # after unpack shape is [k,n]

    # preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm

    # cuda_q_weight_from_preprocessor = preprocessor(unprocessed_weight.cpu(),
    #                              quantized_weight_dtype,
    #                              activation_dtype)

    scale_ref = scale.repeat_interleave(group_size, dim=0)[:k, :]
    ref_th_weight = ref_q_weight.cuda().to(
        activation_dtype
    ) * scale_ref  # convert the weight to FP16 as activation + dequantize the weights

    if has_zero:
        # NOTE needs to change the singture of the kernel, for now we are not using zeros, but needs to pass them
        zero_ref = zero.repeat_interleave(group_size, dim=0)[:k, :]
        # ref_th_weight += zero_ref # NOTE dont needs to add it, change the func signature

    if has_pre_quant:
        pre_quant_scale = pre_quant_scale.repeat(m, 1)
        activation = torch.mul(activation, pre_quant_scale)

    activation = activation.contiguous()
    scale = scale.contiguous()
    zero = zero.contiguous()
    bias = bias.contiguous()
    # Ensure cuda_q_weight for the op is from the unprocessed_weight directly
    cuda_q_weight = unprocessed_weight.cuda().contiguous()

    print(f"activation tensor is {activation}")
    print(f"cuda_q_weight tensor is {cuda_q_weight}")
    print(f"ref_th_weight tensor is {ref_th_weight}")

    output = torch.ops.tensorrt_llm.w4a16_gemm(activation, cuda_q_weight, scale,
                                               zero, bias, group_size)

    # ref is based on ref_th_weight, which in turn is based on the TRT-LLM unpacker op
    ref = _utils.woq_groupwise_gt_matmul(activation,
                                         ref_th_weight.to(activation_dtype),
                                         bias)

    print(f"ref tensor is {ref}")
    print(f"output tensor is {output}")
    # Using a slightly relaxed tolerance for initial testing
    _utils.woq_assert_near_eq(ref, output, 2)


if __name__ == "__main__":
    # Define GEMM dimensions and parameters based on a test case
    m = 8
    n = 64  # Output features. n // 2 must be a multiple of 32 for the preprocessor.
    k = 64  # Inner dimension (input features for the layer), MUST BE A MULTIPLE OF 32
    group_size = 64  # k must be a multiple of group_size for scale_ref.repeat_interleave to work as expected with [:k,:]
    activation_type = torch.float16
    quantized_weight_dtype = torch.quint4x2

    # --- Configuration Flags ---
    HAS_ZERO = True
    HAS_BIAS = True
    has_pre_quant = False  # todo change to true

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_w4a16_gemm_test(m, n, k, group_size, activation_type,
                        quantized_weight_dtype, HAS_ZERO, HAS_BIAS,
                        has_pre_quant, device)
