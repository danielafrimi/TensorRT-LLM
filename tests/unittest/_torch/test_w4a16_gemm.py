import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

from tests.unittest.trt.quantization import _utils


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

    # num_weights_in_32_bits = 8
    # use_int8_weight = 0
    # assert n % num_weights_in_32_bits == 0, f"n must be a multiple of {num_weights_in_32_bits}"

    # shape with int32 now (k, n // 8) -> [64,128]

    unprocessed_weight = torch.randint(
        -128,
        127,
        (
            k, n // 2
        ),  # since the dtype if int32, and each element is 4 bits, we divide by 8.
        dtype=torch.int8,  # does it need to be uint or int?
        device="cuda")
    # shape with int8 is (k, n // 2) -> [64,512]
    # unprocessed_weight = unprocessed_int_weight.view(torch.int8) # unpacks" each int32 into four consecutive int8 values - the shape becomes (k, (n // 8) * 4), which simplifies to (k, n // 2)
    unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
    # Weights must be a CPU Tensor
    ref_q_weight = unpacker(
        unprocessed_weight.cpu())  # after unpack [k,n] -> [64, 1024]
    preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm

    cuda_q_weight = preprocessor(unprocessed_weight.cpu(),
                                 quantized_weight_dtype,
                                 activation_dtype)  # shape is [64, 512]]

    # todo delete it
    aa = torch.tensor([[1, 16]], dtype=torch.int8).reshape(2, 1)
    unpacked_aa = unpacker(aa)
    print(unpacked_aa)

    scale_ref = scale.repeat_interleave(group_size, dim=0)[:k, :]
    ref_th_weight = ref_q_weight.cuda().to(
        activation_dtype
    ) * scale_ref  # convert the weight to FP16 as activation + quantize the weights

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
    cuda_q_weight = cuda_q_weight.cuda().contiguous()

    output = torch.ops.tensorrt_llm.w4a16_gemm(activation, cuda_q_weight, scale,
                                               zero, bias, group_size)

    ref = _utils.woq_groupwise_gt_matmul(activation,
                                         ref_th_weight.to(activation_dtype),
                                         bias)
    _utils.woq_assert_near_eq(ref, output, 2)


if __name__ == "__main__":
    # Define GEMM dimensions and parameters based on a test case
    m = 2
    n = 2  # Output features
    k = 2  # Inner dimension (input features for the layer)
    group_size = 2
    activation_type = torch.float16
    quantized_weight_dtype = torch.quint4x2

    # --- Configuration Flags ---
    HAS_ZERO = True
    HAS_BIAS = True
    has_pre_quant = False
    # Set to True if C++ uses FINEGRAINED_SCALE_AND_ZEROS for the math,
    # False if it uses scale-only math (even if zeros are provided for structure)
    QUANT_MODE_USES_ZEROS = False  # This aligns with the original script's reference calculation: (ref_q_weight_float * scale_ref)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_w4a16_gemm_test(m, n, k, group_size, activation_type,
                        quantized_weight_dtype, HAS_ZERO, HAS_BIAS,
                        has_pre_quant, device)
