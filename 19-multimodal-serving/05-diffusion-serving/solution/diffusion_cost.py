"""What does one image cost? A lower bound from the model's size.

FLUX.1 [dev]: a 12-billion-weight transformer (model card). Its VAE shrinks
each side by 8 and the pipeline packs 2x2 latent cells into one token, so
one image token covers 16x16 pixels (diffusers' pipeline_flux.py). The text
prompt adds up to 512 tokens (max_sequence_length).
"""
from servinglab.cost import hourly_rate
from servinglab.data import hardware

PARAMS = 12e9
STEPS = 50  # num_inference_steps in the model card's example


def tokens(size=1024, text=512):
    side = size // 16
    return side * side + text


def per_step_flops(size=1024):
    return 2 * PARAMS * tokens(size)  # 2 FLOPs per weight per token


def image_seconds(gpu, size=1024, steps=STEPS):
    """Best case: every tensor-core FLOP useful. Real runs take longer."""
    return per_step_flops(size) * steps / (hardware(gpu)["fp16_tflops"] * 1e12)


def usd_per_image(gpu, provider="RunPod", size=1024, steps=STEPS):
    """The floor on what one image costs: GPU price for the best-case seconds."""
    rate, _ = hourly_rate(provider, gpu)
    return rate * image_seconds(gpu, size, steps) / 3600


def intensity(size=1024):
    """FLOPs per byte of weights read in one step (bf16 = 2 bytes)."""
    return per_step_flops(size) / (PARAMS * 2)


if __name__ == "__main__":
    for s in (512, 1024, 2048):
        print(f"{s}x{s}: {tokens(s)} tokens, "
              f"{per_step_flops(s) / 1e12:.0f} TFLOP per step")
    print(f"weights in bf16: {PARAMS * 2 / 1e9:.0f} GB")
    for g in ("T4", "L4", "H100-SXM"):
        h = hardware(g)
        ridge = h["fp16_tflops"] * 1e12 / (h["bandwidth_gbs"] * 1e9)
        print(f"{g:9s} {h['memory_gb']:>3d} GB  ridge {ridge:5.0f} FLOP/B  "
              f"best case {image_seconds(g):6.1f} s per 1024x1024 image")
    print(f"cheapest RunPod H100-SXM row: at least "
          f"${usd_per_image('H100-SXM'):.4f} per image")
    print(f"intensity of one step: {intensity():.0f} FLOP/byte")
