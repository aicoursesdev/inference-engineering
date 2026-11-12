"""Time the slow loop: every new token recomputes the whole sequence.

    uv run python baseline.py
"""

import time

import my_engine
from servinglab.ledger import record

model = my_engine.load("SmolLM2-135M-Instruct")
ids = model.encode("The monsoon reaches Kerala in")
stamps, out = [], list(ids)
start = time.perf_counter()
for token in my_engine.generate(model, ids, 60, temperature=0):
    out.append(token)
    stamps.append(time.perf_counter() - start)

print(model.decode(out[len(ids):]))
gaps = [b - a for a, b in zip([0] + stamps, stamps)]
for n in (0, 9, 29, 59):
    print(f"token {n + 1:2d} (sequence {len(ids) + n:2d} long): "
          f"{gaps[n] * 1000:6.0f} ms")
tps = len(stamps) / stamps[-1]
print(f"{len(stamps)} tokens in {stamps[-1]:.1f} s = {tps:.1f} tokens/s")
record({"lesson": "01.7", "engine": "NumPy engine v0", "model": "SmolLM2-135M",
        "hardware": "M5-Max", "change": "no KV cache, float32",
        "tok_per_s": round(tps, 1), "note": "recompute every token"})
