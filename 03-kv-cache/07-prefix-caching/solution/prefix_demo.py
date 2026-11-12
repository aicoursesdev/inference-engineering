"""Prefix caching: measure what a shared start saves, on the engine and on traces.

    uv run python prefix_demo.py
"""

import time

from servinglab.data import traces
from servinglab.engine import load
from servinglab.kv import PagedCache, replay, step_forward

model = load("SmolLM2-135M-Instruct")
system = model.encode("You are the help desk of Tidewater Goods, a rain-gear shop in Kochi. "
                      "Answer in two short sentences. Never promise a delivery date. " * 3)
asks = [model.encode(q) for q in ("Do you sell kids' raincoats?", "Where is my order 4471?")]
print(f"shared start: {len(system)} tokens, questions: {[len(a) for a in asks]} tokens")

cache = PagedCache(model.config, num_blocks=128, block_size=16)
for i, ask in enumerate(asks):
    prompt = system + ask
    t0 = time.perf_counter()
    view, skipped = cache.start(f"req{i}", prompt)
    step_forward(model, [(view, prompt[skipped:])])
    cache.commit(f"req{i}", prompt)
    ms = (time.perf_counter() - t0) * 1000
    print(f"request {i + 1}: {len(prompt)} tokens, {skipped} found in cache, "
          f"computed {len(prompt) - skipped}, prefill {ms:.0f} ms")
print(f"hit rate so far: {cache.alloc.hit_rate:.0%}")

# On Monsoon's traces: a pool that holds every prefix versus one that is nearly full.
print(f"\n{'trace':10s} {'blocks':>7s} {'no sharing':>11s} {'with sharing':>13s}")
for name, n, blocks in (("tidewater", 600, 2500), ("kodekart", 400, 20000), ("nyaya", 100, 14000)):
    rows = traces(name)[:n]
    a = replay(rows, blocks, "paged")
    b = replay(rows, blocks, "shared")
    print(f"{name:10s} {blocks:7d} {a['hit_rate']:10.0%} {b['hit_rate']:12.0%}  "
          f"(turned away {a['rejected']} -> {b['rejected']})")

# The cache is only as good as the pool that holds it: Nyaya's 15 contracts, 100 questions.
print("\nNyaya, shared prefixes, by pool size")
for blocks in (4000, 14000, 40000):
    r = replay(traces("nyaya"), blocks, "shared")
    print(f"{blocks:6d} blocks = {blocks * 16:7,d} tokens of cache: {r['hit_rate']:.0%} of prompt tokens found")
