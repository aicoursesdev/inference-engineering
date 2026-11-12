"""Overhead you can see on a laptop: a step that does almost no arithmetic.

    uv run python overhead.py

A decode step is dozens of small operations per layer. Each is launched by
Python, and each launch costs a little time whatever the maths inside. The
test model is tiny, so its step time is nearly all launching.
"""

import json
import time
from pathlib import Path

import numpy as np

from servinglab.engine import Config, Model, load
from servinglab.engine.weights import read_safetensors
from servinglab.kv import KVCache, step_forward

TINY = next(p for p in Path(__file__).parents if (p / "servinglab").is_dir()) / "servinglab/tests/data/tiny-llama"
tiny_cfg = Config.from_json(json.loads((TINY / "config.json").read_text()))
tiny = Model(tiny_cfg, read_safetensors(TINY / "model.safetensors"), None, TINY)
small = load("SmolLM2-135M-Instruct")


def decode_seconds(model, context=32, reps=30):
    cache = KVCache(model.config, context + 2)
    step_forward(model, [(cache, list(range(5, 5 + context)))])
    best = 1e9
    for _ in range(reps):
        cache.length = context
        t0 = time.perf_counter()
        step_forward(model, [(cache, [7])])
        best = min(best, time.perf_counter() - t0)
    return best


for name, m in (("tiny test model", tiny), ("SmolLM2-135M", small)):
    c = m.config
    t = decode_seconds(m)
    print(f"{name:16s} {c.layers:2d} layers, hidden {c.hidden:3d}: {t * 1e3:7.2f} ms a step, "
          f"{t / c.layers * 1e6:7.0f} us a layer")

# The cost of one launch: a NumPy call on a tiny array, repeated.
x = np.ones(8, np.float32)
n = 200_000
t0 = time.perf_counter()
for _ in range(n):
    x + x
print(f"\none tiny NumPy call: {(time.perf_counter() - t0) / n * 1e6:.2f} us "
      "(Python and library overhead; the arithmetic is nothing)")
