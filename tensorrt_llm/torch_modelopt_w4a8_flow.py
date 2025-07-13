import torch

group_size = 128
in_features = 128
out_features = 128

quntized_state_dict = torch.load(
    "/home/dafrimi/projects/modelopt_toy_model/pytorch_model.bin")

pre_quant_scale = quntized_state_dict["linear.pre_quant_scale"]
input_scale = quntized_state_dict["linear.input_scale"]
weight_scale = quntized_state_dict["linear.weight_scale"]
weight_scale_2 = quntized_state_dict["linear.weight_scale_2"]
weight = quntized_state_dict["linear.weight"]
# bias = quntized_state_dict["linear.bias"]

model_original_weight = torch.load(
    "/home/dafrimi/projects/modelopt_toy_model/model_weight.pt").to("cuda")
qunatized_weight_before_packing = torch.load(
    "/home/dafrimi/projects/modelopt_toy_model/int8_tensor.pt").to("cuda")
input_tensor = torch.load(
    "/home/dafrimi/projects/modelopt_toy_model/input_tensor.pt").to("cuda")
output_modelopt = torch.load(
    "/home/dafrimi/projects/modelopt_toy_model/output_modelopt.pt").to("cuda")

input_tesor = input_tensor * pre_quant_scale
quantized_input, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
    input_tensor, (input_scale))

unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
weight_unpacked = unpacker(weight.T.to(torch.int8).contiguous().cpu()).T.to(
    input_tensor.dtype).contiguous().cuda()

assert torch.all(
    weight_unpacked.to("cuda") == qunatized_weight_before_packing.to("cuda"))

dequantized_weight = weight_unpacked.to("cuda") * weight_scale.to("cuda")
assert torch.allclose(dequantized_weight,
                      model_original_weight,
                      atol=1e-1,
                      rtol=1e-4)

scale_ref = weight_scale.T.repeat_interleave(group_size, dim=0)
weight_unpacked = weight_unpacked * weight_scale_2

gemm_output = torch.matmul(input_tesor, weight_unpacked)

alpha = (1.0 / (input_scale * weight_scale_2))
output = (gemm_output)


def full_flow():
    input_tensor = input_tensor * pre_quant_scale
    quantized_input, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        input_tensor, (input_scale))  # op divide by input_scale

    weight_unpacked = unpacker(weight.T.to(torch.int8).contiguous().cpu()).T.to(
        input_tensor.dtype).contiguous().cuda()
    fp16_weight = weight_unpacked.to("cuda") * weight_scale.to(
        "cuda")  # int4 to fp16 using weight_Scale
    fp8_weights = fp16_weight / weight_scale_2  # weight_scale_2 is responsible to quanztize the weights from fp16 to fp8 for allow gemm in fp8
    output = quantized_input.to(dtype=fp8_weights.dtype) @ fp8_weights.T
    output = (
        output * weight_scale_2
    ) * input_scale  # because we divide the input by input_scale and weight by weight_scale_2, we need to rescale the output
