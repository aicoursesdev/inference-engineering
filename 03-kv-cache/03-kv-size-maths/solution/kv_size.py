"""How many bytes of keys and values does one token cost? Read it off a config.

    uv run python kv_size.py
"""

from servinglab.data import hardware, model_config
from servinglab.kv import kv_bytes_per_token_from_json

MODELS = ["SmolLM2-135M-Instruct", "Qwen3-0.6B", "Qwen3-4B",
          "Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct"]

print(f"{'model':24s} {'layers':>6s} {'kv heads':>8s} {'head dim':>8s} "
      f"{'bytes/token':>11s} {'4k ctx':>8s} {'128k ctx':>9s}")
for name in MODELS:
    cfg = model_config(name)
    per_token = kv_bytes_per_token_from_json(cfg)           # fp16: 2 bytes a number
    heads = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    print(f"{name:24s} {cfg['num_hidden_layers']:6d} {cfg['num_key_value_heads']:8d} "
          f"{heads:8d} {per_token:11,d} {per_token * 4096 / 2**30:6.2f} GiB "
          f"{per_token * 131072 / 2**30:7.1f} GiB")

# How many 8,000-token chats fit beside Llama 3.1 70B in 8-bit on 8 H100s?
cfg, chip = model_config("Llama-3.1-70B-Instruct"), hardware("H100-SXM")
weights = 70.55e9                                           # 8-bit: one byte a weight
free = 8 * chip["memory_gb"] * 1e9 * 0.9 - weights
per_chat = kv_bytes_per_token_from_json(cfg) * 8000
print(f"\n70B on 8 H100s: {free / 1e9:.0f} GB left for cache, "
      f"{per_chat / 1e9:.2f} GB per 8k chat -> {int(free // per_chat)} chats")
