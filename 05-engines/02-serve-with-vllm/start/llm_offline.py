"""Offline vLLM on a Colab T4 (v0.33.0). Not run here: no NVIDIA GPU."""
from vllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-0.6B", dtype="float16",
          max_model_len=4096, gpu_memory_utilization=0.90)
params = SamplingParams(temperature=0, max_tokens=60)
chat = [{"role": "user",
         "content": "In one sentence: why does my Kochi rain jacket "
                    "leak at the shoulder seams?"}]
out = llm.chat([chat], params,
               chat_template_kwargs={"enable_thinking": False})
print(out[0].outputs[0].text)
