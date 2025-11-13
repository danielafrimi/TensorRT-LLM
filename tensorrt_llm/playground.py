from tensorrt_llm import LLM, SamplingParams
from pathlib import Path


w4a16_model = Path("/home/scratch.dafrimi_gpu/models/kv_cache_quantize/Llama-3.1-8B_qformat_int4_awq_kv_cache_fp8")
fp8_model = Path("/home/scratch.dafrimi_gpu/models/kv_cache_quantize/Llama-3.1-8B_qformat_fp8_kv_cache_fp8")
def main():
    llm = LLM(model=w4a16_model)
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    for output in llm.generate(prompts, sampling_params):
        print(output)



if __name__ == "__main__": 
    main()