"""nano-vllm's block size formula (model_runner.py, allocate_kv_cache),
worked for the course's models. Run from anywhere: uv run python block_bytes.py
"""
from servinglab.data import model_config


def kv_bytes_per_token(name, dtype_bytes=2):
    c = model_config(name)
    head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    # 2 for K and V; one entry per layer, per KV head, per head dimension
    return 2 * c["num_hidden_layers"] * c["num_key_value_heads"] * head_dim * dtype_bytes


if __name__ == "__main__":
    for name in ["Qwen3-0.6B", "Qwen3-4B"]:
        per_tok = kv_bytes_per_token(name)
        print(f"{name}: {per_tok:,} bytes per token, "
              f"{per_tok * 256:,} bytes per 256-token block "
              f"({per_tok * 256 / 2**20:.0f} MiB)")
