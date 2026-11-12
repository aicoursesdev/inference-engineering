"""Pipeline bubbles and replica counts, in numbers.

    uv run python bubbles.py

A pipeline of p stages fed m equal micro-batches is busy m of every (m + p - 1) time slots
(the GPipe schedule; the bubble is the other p - 1 slots). Data parallelism has no bubble
but holds one full copy of the weights per replica.
"""
from servinglab.data import model_config, hardware
from servinglab.roofline import shape

print("pipeline: share of time each stage is idle, (p - 1) / (m + p - 1)")
print("  stages  m=1    m=4    m=16   m=64")
for p in (2, 4, 8):
    print(f"  {p:6}", "  ".join(f"{(p - 1) / (m + p - 1) * 100:4.0f}%" for m in (1, 4, 16, 64)))

print("\ndata parallel replicas of Llama 3.1 8B on 80 GB GPUs (weights BF16, KV left for requests)")
s = shape(model_config("Llama-3.1-8B-Instruct"))
mem = hardware("H100-SXM")["memory_gb"] * 1e9
for replicas in (1, 2, 4, 8):
    kv_room = 0.9 * mem - s["params"] * 2
    print(f"  {replicas} replica(s): weights {replicas * s['params'] * 2 / 1e9:6.1f} GB in total, "
          f"{replicas * kv_room / s['kv_bytes_per_token_f16'] / 1e3:6.0f}k tokens of KV in total")
