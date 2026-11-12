"""Speed and memory of one MLX model: prompt reading, then decoding.

uv run --with mlx-lm python mlx_bench.py mlx-community/Qwen3-0.6B-4bit
"""
import sys

import mlx.core as mx
from mlx_lm import load, stream_generate

repo = sys.argv[1]
model, tokenizer = load(repo)

# A prompt of about 512 tokens so prompt reading is measured for real.
text = "The monsoon reaches Kochi in early June. " * 60
ids = tokenizer.encode(text)[:512]

def run(max_tokens):
    last = None
    for last in stream_generate(model, tokenizer, ids, max_tokens=max_tokens):
        pass
    return last

run(8)  # warm-up: compiles kernels, fills caches
mx.reset_peak_memory()
r = run(128)
print(f"{repo}")
print(f"  prompt  {r.prompt_tokens} tokens at {r.prompt_tps:.0f} tok/s")
print(f"  decode  {r.generation_tokens} tokens at {r.generation_tps:.0f} tok/s")
print(f"  peak memory {r.peak_memory:.2f} GB")
