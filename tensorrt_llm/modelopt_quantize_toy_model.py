import torch

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

export_dir = "/home/dafrimi/projects/modelopt_toy_model"
quntized_state_dict = torch.load(f"{export_dir}/pytorch_model.bin")
dummy = False


def create_identity_matrix(original_weight):
    weight_packed = torch.zeros_like(original_weight, dtype=torch.uint8)
    weight_packed = weight_packed.T
    rows, cols = weight_packed.shape

    min_dim = min(rows, cols * 2)

    for i in range(min_dim):
        col_idx = i // 2
        if col_idx < cols:
            if i % 2 == 0:
                weight_packed[i, col_idx] = 1
            else:
                weight_packed[i, col_idx] = 16

    weight_packed = weight_packed.T
    return weight_packed


def set_params_to_value(original_param, value, dtype):
    return torch.full_like(original_param, value, dtype=dtype)


def create_dummies_linear_inputs(quntized_state_dict):
    linear_inputs = {}

    original_pre_quant_scale = quntized_state_dict["linear.pre_quant_scale"]
    original_weight = quntized_state_dict["linear.weight"]
    original_weight_scale = quntized_state_dict["linear.weight_scale"]
    original_bias = quntized_state_dict["linear.bias"]
    original_input_scale = quntized_state_dict["linear.input_scale"]
    original_weight_scale_2 = quntized_state_dict["linear.weight_scale_2"]

    weight_packed = create_identity_matrix(original_weight)

    linear_inputs["linear.pre_quant_scale"] = torch.ones_like(
        original_pre_quant_scale, dtype=torch.float16)
    linear_inputs["linear.weight"] = weight_packed
    linear_inputs["linear.weight_scale"] = set_params_to_value(
        original_weight_scale, 1.0,
        dtype=torch.float16)  # multiply the weight matrix by scale
    linear_inputs["linear.bias"] = torch.zeros_like(original_bias,
                                                    dtype=torch.float16)
    linear_inputs["linear.input_scale"] = set_params_to_value(
        original_input_scale, 1.0, dtype=torch.float16)  # scale the input
    linear_inputs["linear.weight_scale_2"] = set_params_to_value(
        original_weight_scale_2, 1.0, dtype=torch.float16
    )  # scale the weight matrix by scale look like alpha is * 2

    # Debug: Test unpacking to verify identity matrix
    try:
        unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
        unpacked = unpacker(weight_packed.T.to(
            torch.int8).contiguous().cpu()).contiguous().cuda()
        print(f"\nDebug - Unpacked weight shape: {unpacked.shape}")
        print(
            f"Debug - Number of ones in unpacked weight: {(unpacked == 1).sum().item()}"
        )
        print(f"Debug - Expected number of ones: {min(unpacked.shape)}")

        # Check if it's actually an identity matrix
        min_dim = min(unpacked.shape[0], unpacked.shape[1])
        diagonal_ones = sum(1 for i in range(min_dim) if unpacked[i, i] == 1)
        print(f"Debug - Ones on diagonal: {diagonal_ones}/{min_dim}")

        if diagonal_ones == min_dim:
            print("✓ Successfully created identity matrix!")
        else:
            print("Identity matrix creation may have issues")
    except Exception as e:
        print(f"Debug unpacking failed: {e}")

    return linear_inputs


qc = QuantConfig(quant_algo=QuantAlgo.W4A8_AWQ,
                 group_size=128,
                 has_zero_point=False)

linear_w4a8 = Linear(in_features=128,
                     out_features=128,
                     bias=False,
                     dtype=torch.float16,
                     quant_config=qc)

state_dict = quntized_state_dict if not dummy else create_dummies_linear_inputs(
    quntized_state_dict)

linear_w4a8.load_weights([{
    'pre_quant_scale': (state_dict["linear.pre_quant_scale"]),
    'weight':
    state_dict["linear.weight"],
    'weight_scale':
    state_dict["linear.weight_scale"],
    # 'bias': state_dict["linear.bias"],
    'input_scale':
    state_dict["linear.input_scale"],
    'weight_scale_2':
    state_dict["linear.weight_scale_2"]
}])

linear_w4a8 = linear_w4a8.cuda()

input_tensor = torch.load(f"{export_dir}/input_tensor.pt")
output_modelopt = torch.load(f"{export_dir}/output_modelopt.pt")

output_linear_w4a8 = linear_w4a8(input_tensor.contiguous())
print(f"output_linear_w4a16: {output_linear_w4a8}")
print(f"output_modelopt: {output_modelopt}")
