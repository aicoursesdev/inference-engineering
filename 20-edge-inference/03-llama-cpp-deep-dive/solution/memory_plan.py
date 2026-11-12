"""Will it fit? File size plus the KV cache for the context you ask for.

python memory_plan.py Qwen3-4B 2.49 32768
(model name from data/models, file size in GB, context in tokens)
"""
import sys

from servinglab.data import model_config


def kv_bytes_per_token(cfg, bytes_per_value=2):
    """Keys and values, every layer. 2 bytes each at float16."""
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    return 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * head_dim * bytes_per_value


def plan(model, file_gb, context, bytes_per_value=2):
    kv_gb = kv_bytes_per_token(model_config(model), bytes_per_value) * context / 1e9
    return kv_gb, file_gb + kv_gb


if __name__ == "__main__":
    model, file_gb, context = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    kv, total = plan(model, file_gb, context)
    print(f"{model}: file {file_gb:.2f} GB + KV cache {kv:.2f} GB "
          f"at {context} tokens = {total:.2f} GB")
