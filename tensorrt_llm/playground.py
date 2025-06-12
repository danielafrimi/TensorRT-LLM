from pathlib import Path

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM


def main2():

    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "/dafrimi/projects/models"

    model = AutoModelForCausalLM.from_pretrained(MODEL).to('cuda')
    tok = AutoTokenizer.from_pretrained(MODEL)

    in_tokens = tok("once upon a time", return_tensors="pt").input_ids
    out_tokens = model.generate(in_tokens.to('cuda'))
    print(tok.decode(out_tokens[0]))


def main():
    prompts = [
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is",
        "tell me a story about you"
    ]
    sampling_params = SamplingParams(max_tokens=32)
    # w4a16
    llm = LLM(model=Path('/dafrimi/nvila_quant/NVILA-Lite-8B-INT4/llm'))
    # llm = LLM(model=Path('/dafrimi/Llama-3.1-8B-Instruct_w4a16'))
    #w4a8
    # llm = LLM(model=Path('/dafrimi/projects/Llama-3.1-8B-Instruct_w4a8'))
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
    # main2()
