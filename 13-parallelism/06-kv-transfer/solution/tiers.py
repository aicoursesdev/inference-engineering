"""Load a cached KV from a slower tier, or recompute it? A break-even table.

    uv run python tiers.py

Prefill time is Module 02's roofline ceiling (Llama 3.1 8B, H100), so it is optimistic: a real
prefill is slower, which only makes loading look better. Tier speeds are one-way bandwidths:
PCIe Gen5 x16 is 128 GB/s in both directions together (NVIDIA's H100 page), so 64 GB/s each way;
the NIC is 400 Gb/s = 50 GB/s; "disk" is a made-up 5 GB/s, because drives differ: change it.
"""
from servinglab.roofline import predict

MODEL, CHIP = "Llama-3.1-8B-Instruct", "H100-SXM"
TIERS = [("CPU memory over PCIe", 64e9), ("another machine over a 400 Gb/s NIC", 50e9), ("disk at 5 GB/s", 5e9)]
kv_tok = predict(MODEL, CHIP, context=1)["model"]["kv_bytes_per_token_f16"]
print(f"KV per token: {kv_tok:,} bytes")
print(f"{'tokens':>7} {'KV size':>9} {'recompute':>10}", *[f"{n[:22]:>34}" for n, _ in TIERS])
for n in (500, 3_000, 20_000, 35_000):
    pre = predict(MODEL, CHIP, prompt=n)["prefill"]["seconds"]
    cells = [f"{n * kv_tok / bw * 1e3:7.1f} ms (recompute / load = {pre / (n * kv_tok / bw):4.1f})" for _, bw in TIERS]
    print(f"{n:7} {n * kv_tok / 1e6:7.0f} MB {pre * 1e3:7.1f} ms", *[f"{c:>34}" for c in cells])
