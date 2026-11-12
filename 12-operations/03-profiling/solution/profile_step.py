"""Profile one decode step of a small Llama-shaped model with torch.profiler, on the CPU.

    uv run --with torch python profile_step.py

Random weights, no download: the point is the profile's shape, not the text. One layer
is the real Qwen3-0.6B shape (hidden 1024, 16 query heads, 8 key-value heads, head size
128, MLP 3072), 8 layers of it. Run on any laptop; CUDA-only parts are in the lesson.
"""
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function

torch.manual_seed(0)
H, NQ, NKV, D, I, LAYERS, CTX = 1024, 16, 8, 128, 3072, 8, 512
w = lambda *s: torch.randn(*s) * 0.02
layers = [dict(q=w(H, NQ * D), k=w(H, NKV * D), v=w(H, NKV * D), o=w(NQ * D, H),
               g=w(H, I), u=w(H, I), d=w(I, H)) for _ in range(LAYERS)]
kcache = torch.randn(LAYERS, NKV, CTX, D)
vcache = torch.randn(LAYERS, NKV, CTX, D)


def step(x):                                          # x: (batch, H), one new token each
    for i, p in enumerate(layers):
        with record_function("attention"):
            q = (x @ p["q"]).view(-1, NQ, 1, D)
            k = kcache[i].repeat_interleave(NQ // NKV, 0)[None]   # (1, NQ, CTX, D)
            v = vcache[i].repeat_interleave(NQ // NKV, 0)[None]
            a = F.scaled_dot_product_attention(q, k.expand(q.shape[0], -1, -1, -1), v.expand(q.shape[0], -1, -1, -1))
            x = x + a.reshape(-1, NQ * D) @ p["o"]
        with record_function("mlp"):
            x = x + (F.silu(x @ p["g"]) * (x @ p["u"])) @ p["d"]
    return x


def main():
    for batch in (1, 16):
        x = torch.randn(batch, H)
        step(x)                                           # warm up
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for _ in range(5):
                step(x)
        print(f"\n=== batch {batch}, 5 steps, CPU ===")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=6, max_name_column_width=34))
        if batch == 16:
            prof.export_chrome_trace("decode-step-trace.json")


if __name__ == "__main__":
    main()
