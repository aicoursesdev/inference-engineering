"""Put different chips through the same two questions (Module 02's roofline).

1. Does a 70B FP8 model plus some KV cache fit on one chip?
2. How fast can one request decode, if every token reads all the weights?
Numbers: data/hardware.json for the GPUs (its sources are NVIDIA's H100 and
H200 pages, dense FP8 being the page's "with sparsity" figure halved, as
lesson 02.6 teaches, and AMD's ROCm MI300 page); Google Cloud's TPU7x page
for Ironwood (192 GiB, 7,380 GB/s, 4,614 TFLOPS FP8, not in the json).
MI300X memory: the json holds AMD's 192 GB; ROCm's spec table says 192 GiB
(206 GB). The smaller figure is used here, so its row is the cautious one.
"""
import json
from pathlib import Path

GIB = 2**30
hw = {h["name"]: h for h in json.load(
    open(Path(__file__).resolve().parents[3] / "data" / "hardware.json"))["hardware"]}


def gpu(key):
    h = hw[key]
    return (h["memory_gb"] * 1e9, h["bandwidth_gbs"] * 1e9, h["fp8_tflops"] * 1e12)


chips = {   # name: (memory bytes, bandwidth B/s, dense FP8 FLOP/s)
    "H100 SXM": gpu("H100-SXM"),
    "H200 SXM": gpu("H200"),
    "TPU7x (Ironwood)": (192 * GIB, 7380e9, 4614e12),
    "MI300X": gpu("MI300X"),
}
weights = 70.55e9           # Llama 3.1 70B in FP8, one byte per parameter
kv = 327_680 / 2 * 8192 * 16   # FP8 KV, 16 requests of 8,192 tokens

print(f"weights {weights/1e9:.1f} GB, KV for 16 x 8k tokens {kv/1e9:.1f} GB\n")
print(f"{'chip':18}{'memory':>8}{'free GB':>9}{'KV fits':>9}{'tok/s cap':>10}{'ridge':>7}")
for name, (mem, bw, fl) in chips.items():
    room = (mem * 0.9 - weights) / 1e9      # keep 10% back, as in 17.2
    cap = f"{bw / weights:.0f}" if bw else "n/a"
    ridge = f"{fl / bw:.0f}" if bw and fl else "n/a"
    print(f"{name:18}{mem/1e9:7.0f}G{room:9.1f}{'yes' if room*1e9 >= kv else 'no':>9}"
          f"{cap:>10}{ridge:>7}")
