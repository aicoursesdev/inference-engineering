"""How long does one forward pass take as the sequence grows?

    uv run python forward_cost.py
"""

import time

import my_engine

model = my_engine.load("SmolLM2-135M-Instruct")
ids = list(range(1000, 1256))              # 256 arbitrary token ids
my_engine.forward(model, ids[:8])          # warm up
print("tokens  forward ms  ms per token")
for n in (8, 32, 128, 256):
    start = time.perf_counter()
    my_engine.forward(model, ids[:n])
    ms = (time.perf_counter() - start) * 1000
    print(f"{n:6d}  {ms:10.0f}  {ms / n:12.2f}")

# Work done to write 100 tokens after a 10-token prompt, with and without
# remembering earlier work (token positions pushed through the model):
without = sum(range(10, 110))              # 10, 11, ..., 109
cached = 10 + 99                           # the prompt, then 99 single steps
print(f"no cache: {without:,} positions; remembering: {cached}"
      f" ({without / cached:.0f} times less)")
