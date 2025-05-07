import torch
import torch.nn.functional as F

# Ensure tensorrt_llm is imported to load the custom ops library

# Define GEMM dimensions and parameters based on a test case
M = 1
N_orig = 1024  # Output features
K = 64  # Inner dimension (input features for the layer)
group_size = 64
# Define the target activation type for preprocessing
# For W4A16, the activation type used in the GEMM is FP16
preprocess_activation_type = torch.float16
# Define the weight dtype enum used by the preprocessor and unpacker
# From the test, torch.quint4x2 is used for INT4
weight_dtype_enum = torch.quint4x2  # Used by preprocessor and unpacker

# --- Configuration Flags ---
# Set these based on how the weights were conceptually quantized
# Matches the playground setup and the test case (1, 1024, 64, 'bfloat16', False, True, True, 64) -> using True for has_zero, True for has_bias
HAS_ZERO = True
HAS_BIAS = True
QUANT_MODE_USES_ZEROS = False  # Set to True if C++ uses FINEGRAINED_SCALE_AND_ZEROS - we set to False as we are using scale only

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(
    f"Running W4A16 GEMM on {device} with M={M}, N_orig={N_orig}, K={K}, group_size={group_size}"
)
print(
    f"Reference Config: HAS_ZERO={HAS_ZERO}, HAS_BIAS={HAS_BIAS}, QUANT_MODE_USES_ZEROS={QUANT_MODE_USES_ZEROS}"
)
print(f"Using preprocess_activation_type: {preprocess_activation_type}")
print(f"Using weight_dtype_enum: {weight_dtype_enum}")

# --- Create Input Tensors ---
# 1. Activation Tensor (A)
A = torch.randn(M, K, dtype=torch.float16, device=device)

# 2. "Unprocessed" Packed Weights Tensor (Simulating loaded weights BEFORE preprocessing)
# Created on CPU as the preprocessor/unpacker often expect CPU input
unprocessed_packed_weights_cpu = torch.randint(-128,
                                               127, (K, N_orig // 2),
                                               dtype=torch.int8,
                                               device='cpu')

# --- Preprocess Weights for Custom Op ---
try:
    print("\nPreprocessing weights...")
    # Note: The test file passed unprocessed_weight.cpu()
    # We replicate that here.
    # The view might be incorrect depending on the actual preprocess implementation.
    # performs complex layout transformations (interleaving, swizzling, tiling, padding) on this packed data, optimized
    # for specific CUTLASS kernels. This is its primary purpose, not just packing.
    B_preprocessed = torch.ops.trtllm.preprocess_weights_for_mixed_gemm(
        unprocessed_packed_weights_cpu,  # Input weights on CPU
        weight_dtype_enum,  # Target weight dtype (e.g., torch.quint4x2 for INT4) - can apply the correct layout transformation needed for the corresponding INT4 CUTLASS kernels.
        preprocess_activation_type  # Target activation type (e.g., torch.float16)
    )
    # Move the preprocessed weights to the target device (GPU)
    B_preprocessed = B_preprocessed.to(device)
    print(
        f"Preprocessing successful. Preprocessed weights shape: {B_preprocessed.shape}, dtype: {B_preprocessed.dtype}"
    )
except Exception as e:
    print(f"An error occurred during preprocessing: {e}")
    exit()

# 4. Scales Tensor
scales = torch.rand(K // group_size, N_orig, dtype=torch.float16,
                    device=device) + 0.01

# 5. Zeros Tensor
# Create even if QUANT_MODE_USES_ZEROS is False, as the op signature expects it.
# If HAS_ZERO is False conceptually for the *data*, create a tensor of zeros.
if HAS_ZERO:
    zeros = torch.randn(
        K // group_size, N_orig, dtype=torch.float16, device=device) * 0.1
else:
    zeros = torch.zeros(K // group_size,
                        N_orig,
                        dtype=torch.float16,
                        device=device)

# 6. Bias Tensor
# Create even if HAS_BIAS is False, as the op signature expects it.
# If HAS_BIAS is False, create a tensor of zeros.
if HAS_BIAS:
    bias = torch.randn(N_orig, dtype=torch.float16, device=device)
else:
    bias = torch.zeros(N_orig, dtype=torch.float16, device=device)

# --- Ensure Tensors are Contiguous ---
A = A.contiguous()
B_preprocessed = B_preprocessed.contiguous()
scales = scales.contiguous()
zeros = zeros.contiguous()
bias = bias.contiguous()

# --- Prepare B_packed_final for the custom op call ---
# (Viewing/Reshaping logic as before)
if B_preprocessed.dtype != torch.int8:
    print(
        f"[INFO] Preprocessed weights dtype is {B_preprocessed.dtype}. Viewing as int8 for the GEMM op."
    )
    try:
        # Try viewing raw bytes as int8
        num_int8_elements = B_preprocessed.numel(
        ) * B_preprocessed.element_size()
        B_packed_final = B_preprocessed.view(torch.uint8).view(torch.int8)

        # Attempt to reshape to the expected [K, N_packed] shape.
        # This ASSUMES the preprocessor output total size allows this reshape.
        expected_b_packed_shape = (K, N_orig // 2)
        if B_packed_final.numel() == K * (N_orig // 2):
            B_packed_final = B_packed_final.reshape(
                expected_b_packed_shape).contiguous()
        else:
            print(
                f"[WARNING] Cannot reshape B_packed_final to {expected_b_packed_shape} as number of elements differ."
            )
            # Decide how to proceed - maybe error out or pass the flattened view
            # Passing flattened view if reshape fails:
            # B_packed_final = B_packed_final.contiguous()
            # Let's error out for now if reshape fails after view, as shape is critical
            raise ValueError(
                f"Cannot reshape viewed tensor {B_packed_final.shape} to expected {expected_b_packed_shape}"
            )

    except Exception as e:
        print(
            f"[ERROR] Failed to view/reshape B_preprocessed as int8: {e}. Cannot proceed."
        )
        exit()
else:
    B_packed_final = B_preprocessed.contiguous(
    )  # Ensure contiguous even if already int8

print("\nTensor shapes (Ready for Op Call):")
print(f"  A: {A.shape}, dtype: {A.dtype}")
print(
    f"  B_packed_final: {B_packed_final.shape}, dtype: {B_packed_final.dtype}")
print(f"  scales: {scales.shape}, dtype: {scales.dtype}")
print(f"  zeros: {zeros.shape}, dtype: {zeros.dtype}")
print(f"  bias: {bias.shape}, dtype: {bias.dtype}")
print(f"  group_size: {group_size}")

# --- Execute Custom Operator ---
output_tensor = None
try:
    print("\nCalling torch.ops.tensorrt_llm.w4a16_gemm(...)")
    output_tensor = torch.ops.tensorrt_llm.w4a16_gemm(A, B_packed_final, scales,
                                                      zeros, bias, group_size)
    print("Custom op call successful!")
    print(f"Output tensor shape: {output_tensor.shape}")
    print(f"Output tensor dtype: {output_tensor.dtype}")
except Exception as e:
    print(f"An error occurred during custom op call: {e}")
    # Don't exit immediately, try to compute reference if possible

# --- Reference Calculation ---
print("\nCalculating reference output...")
output_ref = None
try:
    # 1. Unpack weights (CPU tensor expected by unpacker)
    # Needs shape [N_orig/2, K] potentially? Check unpacker requirements.
    # The test file passes [K, N_orig//8] viewed as int8. Let's try that structure.
    # Input to unpacker should be int8 view of int32 data [K, N_orig//8] ??? This is confusing.
    # Let's stick to the test file's pattern: use the cpu int8 tensor we created initially.
    ref_unpacked_q_weight_int8 = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8(
        unprocessed_packed_weights_cpu)  # Shape [K, N_orig]
    ref_unpacked_q_weight_int8 = ref_unpacked_q_weight_int8.to(
        device)  # Move to GPU
    print(
        f"  Reference unpacked weights shape: {ref_unpacked_q_weight_int8.shape}, dtype: {ref_unpacked_q_weight_int8.dtype}"
    )

    # 2. Prepare scales and zeros by repeating rows
    scale_ref = scales.repeat_interleave(group_size,
                                         dim=0)  # Shape becomes [K, N_orig]
    zero_ref = zeros.repeat_interleave(group_size,
                                       dim=0)  # Shape becomes [K, N_orig]
    print(f"  Reference repeated scales shape: {scale_ref.shape}")
    print(f"  Reference repeated zeros shape: {zero_ref.shape}")

    # Ensure shapes are compatible if K was not perfectly divisible by group_size originally
    # (Our example has K=64, group_size=64, so K_final = K)
    K_final = ref_unpacked_q_weight_int8.shape[0]
    scale_ref = scale_ref[:K_final, :]
    zero_ref = zero_ref[:K_final, :]

    # 3. Dequantize weights
    # Formula depends on whether zeros are used in the *actual* quantization scheme
    ref_q_weight_float = ref_unpacked_q_weight_int8.to(
        torch.float16)  # Convert int8 unpacked weights to float16

    if QUANT_MODE_USES_ZEROS:  # If the math should use zeros
        ref_dequant_weight = (ref_q_weight_float - zero_ref) * scale_ref
        print("  Dequantizing using: (int_weight - zero) * scale")
    else:  # If scale-only math
        ref_dequant_weight = ref_q_weight_float * scale_ref
        print("  Dequantizing using: int_weight * scale")
        if HAS_ZERO:  # If data conceptually had zeros but math ignores them
            print(
                "  (Note: Zeros tensor was provided but ignored in scale-only reference math)"
            )

    # 4. Perform GEMM: A[M, K] @ W_dequant[K, N] -> C[M, N]
    # Need to transpose W if it's [K, N] -> [N, K] for matmul, or use F.linear
    # F.linear(input[..., K], weight[N, K]) -> output[..., N]
    output_ref = F.linear(
        A, ref_dequant_weight.t())  # Transpose W from [K,N] to [N,K]
    print(f"  Reference GEMM output shape: {output_ref.shape}")

    # 5. Add Bias
    if HAS_BIAS:
        output_ref = output_ref + bias  # Bias shape [N] should broadcast
        print("  Added bias to reference output.")

    print("Reference calculation successful!")

except Exception as e:
    print(f"An error occurred during reference calculation: {e}")

# --- Compare Results ---
if output_tensor is not None and output_ref is not None:
    print("\nComparing outputs...")
    try:
        # Tolerances might need adjustment based on quantization effects
        atol = 1e-1
        rtol = 1e-2
        are_close = torch.allclose(output_tensor,
                                   output_ref,
                                   atol=atol,
                                   rtol=rtol)
        if are_close:
            print(f"Outputs match within tolerance (atol={atol}, rtol={rtol})!")
        else:
            print("Outputs DO NOT match within tolerance.")
            diff = torch.abs(output_tensor - output_ref)
            print(f"  Max difference: {torch.max(diff)}")
            # print("  Output Custom Op (first 5 flattened):", output_tensor.flatten()[:5])
            # print("  Output Reference (first 5 flattened):", output_ref.flatten()[:5])
    except Exception as e:
        print(f"An error occurred during comparison: {e}")
elif output_tensor is None:
    print("\nSkipping comparison because custom op failed.")
else:
    print("\nSkipping comparison because reference calculation failed.")
