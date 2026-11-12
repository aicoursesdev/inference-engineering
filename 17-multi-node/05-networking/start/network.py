"""Three network bills: TP all-reduce, KV transfer, MoE all-to-all.

Bandwidth-only floors: real collectives add latency and never reach line rate.
Link speeds (published): NVLink 4 on H100/H200 = 900 GB/s total per GPU, so
450 GB/s each way (NVIDIA H100 page); InfiniBand NDR = 400 Gb/s = 50 GB/s
(Perplexity's EFA post). 359 Gbps was the best llm-d saw on one ConnectX-7.
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data" / "models"
NVLINK = 450e9      # bytes/s, one direction
IB = 50e9           # bytes/s, one 400 Gb/s NIC
ib_seen = 359e9 / 8

llama = json.load(open(DATA / "Llama-3.1-70B-Instruct.json"))
L, H = llama["num_hidden_layers"], llama["hidden_size"]

# 1. Tensor parallelism: two all-reduces per layer, each over B x H values.
print("1. TP=8 all-reduce per decode step, Llama 70B, BF16")
for B in (32, 256):
    size = B * H * 2
    ring = 2 * (8 - 1) / 8 * size            # bytes each GPU moves
    n = 2 * L
    for name, bw in [("NVLink", NVLINK), ("InfiniBand", IB)]:
        t = ring / bw
        print(f"  batch {B:3}: {size/1e6:5.2f} MB message, {name:10} "
              f"{t*1e6:6.0f} us each x {n} = {t*n*1e3:5.1f} ms per step")

# 2. KV transfer from a prefill node to a decode node.
per_tok = 2 * L * llama["num_key_value_heads"] * (H // llama["num_attention_heads"]) * 2
print(f"\n2. KV transfer, Llama 70B BF16, {per_tok:,} bytes per token")
for who, toks in [("Tidewater chat", 600), ("Kodekart step", 20000),
                  ("Nyaya contract", 40000)]:
    size = per_tok * toks
    print(f"  {who:15}{toks:6} tokens = {size/1e9:5.2f} GB: "
          f"{size/IB*1e3:6.0f} ms on one 50 GB/s NIC, "
          f"{size/(8*IB)*1e3:5.0f} ms if 8 GPUs each use their own NIC")

oss = json.load(open(DATA / "gpt-oss-120b.json"))
d = oss["head_dim"]; kvh = oss["num_key_value_heads"]
full = oss["layer_types"].count("full_attention")
slide = oss["layer_types"].count("sliding_attention")
toks = 5000
size = 2 * kvh * d * 2 * (full * toks + slide * min(toks, oss["sliding_window"]))
print(f"\n   llm-d's test: gpt-oss-120b, {toks}-token prompt = {size/1e6:.0f} MB "
      f"in total, {size/4/1e6:.0f} MB per TP=4 shard (llm-d reports ~48 MB)")
print(f"   one ConnectX-7 at the 359 Gbps llm-d saw: "
      f"{ib_seen/1e9:.1f} GB/s -> {48e6/ib_seen*1e3:.2f} ms for 48 MB at best")

# 3. MoE all-to-all, DeepSeek-V3 decode (FP8 dispatch, BF16 combine)
ds = json.load(open(DATA / "DeepSeek-V3.json"))
h, k = ds["hidden_size"], ds["num_experts_per_tok"]
tokens = 128                                  # per GPU, as in Perplexity's test
disp, comb = tokens * h * 1 * k, tokens * h * 2 * k
moe = ds["num_hidden_layers"] - ds["first_k_dense_replace"]
print(f"\n3. MoE all-to-all, {tokens} tokens per GPU, top-{k}, hidden {h}")
print(f"   dispatch {disp/1e6:.1f} MB + combine {comb/1e6:.1f} MB per layer")
t = (disp + comb) / IB
print(f"   all remote over one 50 GB/s NIC: {t*1e6:.0f} us per layer, "
      f"{t*moe*1e3:.0f} ms over {moe} layers")
t = (disp + comb) / NVLINK
print(f"   all inside one NVLink domain:    {t*1e6:.0f} us per layer, "
      f"{t*moe*1e3:.1f} ms over {moe} layers")
