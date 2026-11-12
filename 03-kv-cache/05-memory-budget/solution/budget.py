"""Split a GPU's memory into weights, activations and cache, and predict an OOM.

    uv run python budget.py
"""

from servinglab.data import hardware, model_config
from servinglab.engine import load
from servinglab.kv import KVCache, kv_bytes_per_token_from_json, memory_budget

# 1. The formula against real arrays: SmolLM2-135M's weights and a cache.
model = load("SmolLM2-135M-Instruct")
cache = KVCache(model.config, 2048)
per_token = kv_bytes_per_token_from_json(model_config("SmolLM2-135M-Instruct"), 4)  # float32
print(f"weights in memory: {sum(w.nbytes for w in model.weights.values()) / 1e6:6.1f} MB "
      f"({model.n_params / 1e6:.1f}M parameters x 4 bytes)")
print(f"cache, 2048 slots: {cache.k.nbytes / 1e6 + cache.v.nbytes / 1e6:6.1f} MB "
      f"(formula: {per_token * 2048 / 1e6:.1f} MB)")

# 2. Qwen3-4B on three GPUs, 16-bit weights, a 90% cap.
print(f"\n{'GPU':10s} {'usable':>7s} {'weights':>8s} {'activ.':>7s} {'other':>6s} "
      f"{'cache':>7s} {'tokens':>8s} {'8k chats':>9s}")
for gpu in ("T4", "L4", "A100-40GB"):
    b = memory_budget("Qwen3-4B", gpu)
    print(f"{gpu:10s} {b['usable'] / 1e9:6.1f}G {b['weights'] / 1e9:7.2f}G "
          f"{b['activations'] / 1e9:6.2f}G {b['overhead'] / 1e9:5.1f}G "
          f"{b['kv'] / 1e9:6.2f}G {b['kv_tokens']:8,d} {b['kv_tokens'] // 8000:9d}")

# 3. Will it fit? Llama 3.1 8B on a T4 (16 GB): 16-bit weights, then 8-bit.
for bits, bpp in ((16, 2), (8, 1)):
    b = memory_budget("Llama-3.1-8B-Instruct", "T4", bytes_per_param=bpp)
    verdict = f"{b['kv_tokens']:,d} tokens of cache" if b["kv"] > 0 else "out of memory before any cache"
    print(f"\nLlama 3.1 8B, {bits}-bit weights, T4: weights {b['weights'] / 1e9:.1f} GB -> {verdict}")
