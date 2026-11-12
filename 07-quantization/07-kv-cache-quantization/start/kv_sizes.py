"""KV cache against weights: when does the cache outweigh the model?

    uv run python kv_sizes.py

KV bytes per token = 2 (K and V) x layers x KV heads x head size x bytes per
number. llama.cpp's q8_0 stores 8.5 bits per number and q4_0 4.5 bits (32
values plus a 16-bit scale per block).
"""
from servinglab.data import model_config

BITS = {"fp16": 16, "q8_0": 8.5, "q4_0": 4.5}
WEIGHTS_GB = {"Qwen3-0.6B": 1.19, "Qwen3-4B": 8.04, "Llama-3.1-8B-Instruct": 16.06}


def per_token(name, bits):
    c = model_config(name)
    hd = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    return 2 * c["num_hidden_layers"] * c["num_key_value_heads"] * hd * bits / 8


if __name__ == "__main__":
    for name, w in WEIGHTS_GB.items():
        print(f"{name} (weights {w} GB at 16 bits)")
        for fmt, bits in BITS.items():
            pt = per_token(name, bits)
            parity = w * 1e9 / pt
            print(f"  KV {fmt:5} {pt:9,.0f} B/token   32k tokens {pt * 32768 / 1e9:5.2f} GB"
                  f"   cache = weights at {parity:9,.0f} tokens")
