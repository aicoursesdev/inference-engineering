"""Time generation with and without the KV cache.

    uv run python time_cache.py
"""

import statistics
import time

from my_cache import generate_cached
from servinglab.engine import generate, load
from servinglab.ledger import record

model = load("SmolLM2-135M-Instruct")
ids = model.encode("The monsoon reaches Kerala in")
N = 60


def timed(gen):
    """Seconds since the start at which each new token appeared."""
    start, stamps = time.perf_counter(), []
    for _ in gen:
        stamps.append(time.perf_counter() - start)
    return stamps


runs = {"no cache": [], "cache": []}
for _ in range(3):                                   # three runs each, the middle one counts
    runs["no cache"].append(timed(generate(model, ids, N, temperature=0)))
    runs["cache"].append(timed(generate_cached(model, ids, N, temperature=0)))

best = {}
for name, trials in runs.items():
    s = sorted(trials, key=lambda t: t[-1])[1]       # the median run
    gaps = [b - a for a, b in zip([0] + s, s)]
    best[name] = s
    print(f"{name:9s} first token {gaps[0] * 1000:5.0f} ms | "
          f"token {N} {gaps[-1] * 1000:5.0f} ms | "
          f"{N} tokens in {s[-1]:5.2f} s = {N / s[-1]:5.1f} tok/s")
speedup = best["no cache"][-1] / best["cache"][-1]
print(f"speedup {speedup:.1f}x")
record({"lesson": "03.2", "engine": "NumPy engine + KV cache", "model": "SmolLM2-135M",
        "hardware": "M5-Max", "change": "add a KV cache", "tok_per_s": round(N / best["cache"][-1], 1),
        "note": f"{speedup:.1f}x faster than recomputing"})
