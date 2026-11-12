"""How much does one decode step cost as more requests share it?

    uv run python batch_curve.py

Real measurement on the NumPy engine: every request has 64 tokens of context
and decodes one token. The weights are read once per step whatever the
batch, so tokens per second rise with it.
"""

import statistics
import time

from servinglab.engine import load
from servinglab.kv import KVCache, step_forward

model = load("SmolLM2-135M-Instruct")
ids = list(range(100, 164))
print(f"{'batch':>5s} {'step':>9s} {'tokens/s':>9s} {'per request':>12s}")
base = None
for batch in (1, 2, 4, 8, 16, 32):
    caches = []
    for _ in range(batch):
        c = KVCache(model.config, 128)
        step_forward(model, [(c, ids)])                      # prefill once
        caches.append(c)
    times = []
    for rep in range(5):
        for c in caches:
            c.length = 64                                    # rewind: decode the same slot
        t0 = time.perf_counter()
        step_forward(model, [(c, [7]) for c in caches])
        times.append(time.perf_counter() - t0)
    best = min(times)                                        # least disturbed by other programs
    base = base or best
    print(f"{batch:5d} {best * 1000:7.1f} ms {batch / best:9.0f} {best / base:11.1f}x")
