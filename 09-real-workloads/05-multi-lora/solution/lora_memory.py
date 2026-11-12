"""How big is a LoRA adapter, and how many fit next to one base model?

A LoRA adapter replaces the change to a weight matrix W (d_out x d_in) by two
thin matrices, A (r x d_in) and B (d_out x r), so it stores r * (d_in + d_out)
numbers instead of d_in * d_out. Here it is applied to all seven matrices of
every layer (q, k, v, o, gate, up, down), the common choice.

    uv run python lora_memory.py
"""

from servinglab.data import model_config


def adapter_params(cfg, rank):
    h, layers = cfg["hidden_size"], cfg["num_hidden_layers"]
    d = cfg.get("head_dim") or h // cfg["num_attention_heads"]
    q_out, kv_out = cfg["num_attention_heads"] * d, cfg["num_key_value_heads"] * d
    ff = cfg["intermediate_size"]
    shapes = [(h, q_out), (h, kv_out), (h, kv_out), (q_out, h),   # q k v o
              (h, ff), (h, ff), (ff, h)]                          # gate up down
    return layers * sum(rank * (d_in + d_out) for d_in, d_out in shapes)


print(f"{'model':24} {'rank':>4} {'adapter':>10} "
      f"{'100 adapters':>13}")
for name in ("Qwen3-4B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct"):
    cfg = model_config(name)
    for rank in (8, 16, 64):
        n = adapter_params(cfg, rank)
        mb = n * 2 / 1e6                           # bfloat16: 2 bytes each
        print(f"{name:24} {rank:4d} {mb:8.0f} MB "
              f"{100 * mb / 1000:10.1f} GB")
