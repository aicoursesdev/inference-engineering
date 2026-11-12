"""Count the cache each attention design needs, from real configs.

    uv run python shrink.py
"""

from servinglab.data import model_config

GIB = 2 ** 30


def per_token(layers, kv_heads, head_dim, bytes_per=2):
    return 2 * layers * kv_heads * head_dim * bytes_per


# 1. The same Llama 3.1 8B shape with 32, 8 and 1 key/value heads.
cfg = model_config("Llama-3.1-8B-Instruct")
L, H, D = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["head_dim"] if "head_dim" in cfg else 128
for name, kv in (("MHA", H), ("GQA", cfg["num_key_value_heads"]), ("MQA", 1)):
    b = per_token(L, kv, D)
    print(f"{name} ({kv:2d} kv heads): {b:9,d} B/token = {b * 131072 / GIB:5.1f} GiB at 128k")

# 2. DeepSeek-V3's latent attention keeps one small vector per token and layer.
ds = model_config("DeepSeek-V3")
latent = ds["kv_lora_rank"] + ds["qk_rope_head_dim"]
mla = ds["num_hidden_layers"] * latent * 2
# What plain multi-head attention would store: every head's key and value.
mha = (ds["num_hidden_layers"] * ds["num_attention_heads"] * 2        # 2 bytes a number
       * (ds["qk_nope_head_dim"] + ds["qk_rope_head_dim"] + ds["v_head_dim"]))
print(f"\nDeepSeek-V3: latent {ds['kv_lora_rank']} + rope {ds['qk_rope_head_dim']} "
      f"= {latent} numbers a layer -> {mla:,d} B/token "
      f"({mla * 131072 / GIB:.1f} GiB at 128k)")
print(f"  plain multi-head would need {mha:,d} B/token "
      f"({mha * 131072 / GIB:.0f} GiB at 128k): {mha / mla:.0f}x more")

# 3. gpt-oss-20b: half its layers see only the last `sliding_window` tokens.
g = model_config("gpt-oss-20b")
layers, kv, d, w = (g["num_hidden_layers"], g["num_key_value_heads"],
                    g["head_dim"], g["sliding_window"])
full = sum(t == "full_attention" for t in g["layer_types"])
for ctx in (4096, 131072):
    dense = per_token(layers, kv, d) * ctx
    banded = per_token(full, kv, d) * ctx + per_token(layers - full, kv, d) * min(ctx, w)
    print(f"gpt-oss-20b at {ctx:6,d} tokens: all-dense {dense / 2**20:7.0f} MiB, "
          f"with windows {banded / 2**20:7.0f} MiB ({banded / dense:.0%})")
