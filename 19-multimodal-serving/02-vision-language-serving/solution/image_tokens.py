"""How many tokens does an image add, and what does that cost to read?

Qwen2.5-VL and Qwen3-VL cut an image into square patches, merge each
2x2 block of patches into one token, and clamp the size first.
Numbers come from each model's own preprocessor_config.json.
"""
import math
from dataclasses import dataclass


@dataclass
class Vision:
    name: str
    patch: int        # pixels per patch side
    merge: int        # patches merged per side into one token
    min_pixels: int
    max_pixels: int

    @property
    def factor(self):  # pixels per token side
        return self.patch * self.merge


QWEN25 = Vision("Qwen2.5-VL-7B", 14, 2, 3136, 12845056)
QWEN3 = Vision("Qwen3-VL-8B", 16, 2, 65536, 16777216)


def smart_resize(h, w, v):
    """The resize rule of Hugging Face's Qwen2-VL image processor."""
    f = v.factor
    hb, wb = round(h / f) * f, round(w / f) * f
    if hb * wb > v.max_pixels:
        beta = math.sqrt(h * w / v.max_pixels)
        hb = max(f, math.floor(h / beta / f) * f)
        wb = max(f, math.floor(w / beta / f) * f)
    elif hb * wb < v.min_pixels:
        beta = math.sqrt(v.min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
    return hb, wb


def image_tokens(h, w, v):
    hb, wb = smart_resize(h, w, v)
    return (hb // v.factor) * (wb // v.factor)


def kv_bytes_per_token(cfg):
    t = cfg.get("text_config", cfg)
    head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    # keys and values, every layer, 2 bytes each (float16)
    return 2 * t["num_hidden_layers"] * t["num_key_value_heads"] * head_dim * 2


def matmul_params(cfg):
    """Weights every token is multiplied by (no embedding table)."""
    t = cfg.get("text_config", cfg)
    d, L = t["hidden_size"], t["num_hidden_layers"]
    hd = t.get("head_dim") or d // t["num_attention_heads"]
    attn = d * hd * (t["num_attention_heads"] + 2 * t["num_key_value_heads"])
    attn += hd * t["num_attention_heads"] * d
    mlp = 3 * d * t["intermediate_size"]
    return L * (attn + mlp)


def prefill_seconds(tokens, cfg, gpu):
    """Best case: the matrix maths only, at the chip's dense peak.
    Real runs are slower, and the vision encoder adds more on top."""
    from servinglab.data import hardware
    flops = 2 * matmul_params(cfg) * tokens
    return flops / (hardware(gpu)["fp16_tflops"] * 1e12)


if __name__ == "__main__":
    import json
    cfg = json.load(open("qwen3_vl_8b_config.json"))
    for h, w, label in [(224, 224, "thumbnail"), (1080, 1920, "screenshot"),
                        (3000, 4000, "phone photo")]:
        a, b = image_tokens(h, w, QWEN25), image_tokens(h, w, QWEN3)
        print(f"{label:11s} {w}x{h}: Qwen2.5-VL {a:5d}  Qwen3-VL {b:5d}")
    print("KV bytes per token:", kv_bytes_per_token(cfg))
    print("matmul params:", matmul_params(cfg) / 1e9, "B")
    n = image_tokens(1080, 1920, QWEN3)
    print(f"{n} tokens: KV {n * kv_bytes_per_token(cfg) / 1e6:.0f} MB")
    for gpu in ("T4", "L4", "H100-SXM"):
        print(f"{gpu:9s} prefill at least {prefill_seconds(n, cfg, gpu):.3f} s")
