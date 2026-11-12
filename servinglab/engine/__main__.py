"""python -m servinglab.engine "your prompt": generate and print the numbers."""

import sys
import time

from . import generate, load


def main():
    prompt = " ".join(sys.argv[1:]) or "The capital of France is"
    model = load("SmolLM2-135M-Instruct")
    c = model.config
    print(f"{model.n_params:,} parameters, "
          f"{model.n_bytes_bf16 / 1e6:.0f} MB as bfloat16, "
          f"{c.layers} layers")
    ids = model.encode(prompt)
    start = time.perf_counter()
    out = list(generate(model, ids, max_new_tokens=30, temperature=0))
    secs = time.perf_counter() - start
    print(prompt + model.decode(out))
    print(f"{len(out)} tokens in {secs:.1f} s = {len(out) / secs:.1f} tokens/s")


main()
