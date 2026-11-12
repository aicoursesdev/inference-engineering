"""Will it fit? The sums behind vLLM's memory settings.

    uv run python memory_budget.py Qwen3-4B T4 0.92 4096

Prints weights, the KV cache that is left and what it holds. It is an upper
bound: vLLM also subtracts the memory it measures for activations at start.
"""
import sys

from servinglab.data import hardware, model_config

GB = 1e9


def param_count(c):
    """Parameters of a Qwen3-style dense model, from its config."""
    d, hd = c["hidden_size"], c["head_dim"]
    q, kv = c["num_attention_heads"] * hd, c["num_key_value_heads"] * hd
    attn = d * q + 2 * d * kv + q * d
    mlp = 3 * d * c["intermediate_size"]
    embed = c["vocab_size"] * d          # tied: the output head shares it
    return c["num_hidden_layers"] * (attn + mlp) + embed


def budget(model, gpu, fraction, max_len, dtype_bytes=2):
    c, g = model_config(model), hardware(gpu)
    weights = param_count(c) * dtype_bytes
    per_token = (2 * c["num_hidden_layers"] * c["num_key_value_heads"]
                 * c["head_dim"] * dtype_bytes)
    kv = g["memory_gb"] * GB * fraction - weights
    return {"params_b": param_count(c) / GB, "weights_gb": weights / GB,
            "kv_gb": kv / GB, "bytes_per_token": per_token,
            "tokens_held": int(kv // per_token),
            "one_request_gb": max_len * per_token / GB,
            "full_length_requests": kv / (max_len * per_token)}


if __name__ == "__main__":
    model, gpu, fraction, max_len = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    for k, v in budget(model, gpu, fraction, max_len).items():
        print(f"{k:22} {v:,.2f}")
