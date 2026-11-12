"""Print the engine's size and speed, and add the first engine row to the ledger.

    uv run python report.py
"""

import time

import engine
from servinglab.ledger import record

model = engine.load("SmolLM2-135M-Instruct")
n = model.n_params
print(f"parameters:   {n:,}")
print(f"weight bytes: {2 * n:,} as bfloat16 ({2 * n / 1e6:.0f} MB)")

ids = model.encode("The capital of France is")
start = time.perf_counter()
out = list(engine.generate(model, ids, 30, temperature=0))
secs = time.perf_counter() - start
print(model.decode(ids + out))
print(f"{len(out)} new tokens in {secs:.1f} s = {len(out) / secs:.1f} tokens/s")
record({"lesson": "01.8", "engine": "NumPy engine v0", "model": "SmolLM2-135M",
        "hardware": "M5-Max", "change": "engine v0 report, no KV cache",
        "tok_per_s": round(len(out) / secs, 1),
        "note": f"{n:,} parameters"})
