"""Blocks and block tables, on the real engine and on a trace.

    uv run python paged_demo.py
"""

import numpy as np

from servinglab.data import traces
from servinglab.engine import load
from servinglab.kv import KVCache, PagedCache, forward_cached, replay, step_forward

model = load("SmolLM2-135M-Instruct")
ids = model.encode("The monsoon reaches Kerala in the first week of June, "
                   "and Kochi gets the first heavy rain.")

# 1. A sequence in blocks of 8 tokens: its block table, in the order the pool gave them out.
cache = PagedCache(model.config, num_blocks=32, block_size=8)
cache.alloc.allocate("filler", list(range(21)))            # someone else holds 3 blocks first
view, _ = cache.start("a", ids)
logits = step_forward(model, [(view, ids)])[0]
print(f"{len(ids)} tokens, block size 8 -> block table {cache.alloc.seqs['a'].table}")

# 2. Same answer as a contiguous cache, though the blocks are scattered.
flat = forward_cached(model, ids, KVCache(model.config, 64))
print("same logits as the contiguous cache:", bool(np.allclose(logits, flat, atol=1e-4)))

# 3. Waste on Tidewater's chat traffic: reserve 2,048 slots a request, or use blocks.
rows = traces("tidewater")[:400]
for mode in ("reserve", "paged"):
    r = replay(rows, num_blocks=4000, mode=mode, max_len=2048)
    print(f"{mode:8s} waste {r['mean_waste']:5.1%}, turned away {r['rejected']}")
