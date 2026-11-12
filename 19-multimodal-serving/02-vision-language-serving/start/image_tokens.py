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
    raise NotImplementedError("TODO 19.2: round each side to a multiple of v.factor, then shrink (if over max_pixels) or grow (if under min_pixels) as the Hugging Face rule does, see the lesson's 'How an image becomes tokens' section")


def image_tokens(h, w, v):
    raise NotImplementedError("TODO 19.2: resized height / factor times resized width / factor, see the lesson's 'How an image becomes tokens' section")


def kv_bytes_per_token(cfg):
    raise NotImplementedError("TODO 19.2: 2 (keys and values) x layers x kv heads x head_dim x 2 bytes, see the lesson's 'What those tokens cost' section")


def matmul_params(cfg):
    """Weights every token is multiplied by (no embedding table)."""
    raise NotImplementedError("TODO 19.2: per layer: the attention q, k, v, o matrices plus the three MLP matrices, times layers, see the lesson's 'What those tokens cost' section")


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
