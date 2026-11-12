"""Is it faster to reload a saved cache or to recompute it? The break-even.

    uv run python tier_math.py

The link speeds are choices to try, not measurements: put in your own
machine's number (a CPU-to-GPU copy you timed, or a network or disk speed).
"""

from servinglab.data import model_config
from servinglab.kv import kv_bytes_per_token_from_json
from servinglab.roofline import predict

MODEL, GPU = "Qwen3-4B", "L4"
per_token = kv_bytes_per_token_from_json(model_config(MODEL))
print(f"{MODEL} on {GPU}: {per_token:,d} bytes of cache per token\n")
print(f"{'tokens':>8s} {'cache':>8s} {'recompute':>10s} | seconds to reload at:"
      f" {'2 GB/s':>7s} {'10 GB/s':>8s} {'50 GB/s':>8s}")
for n in (2_000, 8_000, 32_000, 128_000):
    recompute = predict(MODEL, GPU, prompt=n)["prefill"]["seconds"]
    size = per_token * n
    cells = " ".join(f"{size / (gbs * 1e9):8.2f}" for gbs in (2, 10, 50))
    print(f"{n:8,d} {size / 1e9:7.2f}G {recompute:9.2f}s | {cells}")
