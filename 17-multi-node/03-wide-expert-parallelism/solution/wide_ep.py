"""Check DeepSeek's published numbers, then ask what wide EP buys."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data" / "models"
cfg = json.load(open(DATA / "DeepSeek-V3.json"))
routed = cfg["n_routed_experts"]          # 256
active = cfg["num_experts_per_tok"]       # 8
redundant = 32                            # DeepSeek's overview, both phases

print(f"each token uses {active} of {routed} routed experts")
for name, gpus in [("DeepSeek prefill EP32", 32), ("DeepSeek decode EP144", 144),
                   ("LMSYS decode EP72", 72)]:
    per_gpu = (routed + redundant) / gpus
    print(f"{name}: {routed}+{redundant} experts / {gpus} GPUs "
          f"= {per_gpu:g} routed experts per GPU")

# DeepSeek's 24-hour accounting (Feb 27-28, 2025), numbers from its page
nodes, gpus_per_node, rate, hours = 226.75, 8, 2.0, 24
cost = nodes * gpus_per_node * rate * hours
print(f"\ncost: {nodes} nodes x {gpus_per_node} GPUs x ${rate}/h x {hours} h "
      f"= ${cost:,.0f}   (page: $87,072)")

hit, total_in, out = 342e9, 608e9, 168e9
revenue = (hit * 0.14 + (total_in - hit) * 0.55 + out * 2.19) / 1e6
print(f"revenue at R1 prices: ${revenue:,.0f}   (page: $562,027)")
print(f"margin: {(revenue - cost) / cost:.0%}   (page: 545%)")

# how many tokens did one node handle per second, on average?
print(f"\noutput tokens per node-second: {out / (nodes * 86400):,.0f}")
print(f"input tokens per node-second:  {total_in / (nodes * 86400):,.0f}")
print("(averages over all nodes, prefill and decode together; the page's "
      "73.7k and 14.8k are per busy node in each phase)")

# LMSYS vs DeepSeek decode, per node, from the LMSYS post
print(f"\nLMSYS 22,282 vs DeepSeek blog 14,800 output tokens/s per node: "
      f"{22282 / 14800:.2f}x (different batch, KV length and node count)")

# Expert weights held per GPU (FP8, one byte per parameter): the memory
# that wide EP hands back to the KV cache. One expert = gate, up, down.
h, ff = cfg["hidden_size"], cfg["moe_intermediate_size"]
moe_layers = cfg["num_hidden_layers"] - cfg["first_k_dense_replace"]
one_expert_all_layers = 3 * h * ff * moe_layers
print(f"\none expert across {moe_layers} MoE layers: "
      f"{one_expert_all_layers / 1e9:.2f}B params = "
      f"{one_expert_all_layers / 1e9:.2f} GB in FP8")
for name, per_gpu in [("prefill, 9 per GPU", 9), ("decode, 2 per GPU", 2)]:
    print(f"  {name}: {per_gpu * one_expert_all_layers / 1e9:.1f} GB of experts")

# Tokens each expert sees per decode step, if routing were perfectly even.
# Every GPU holds `per_gpu` sequences; each token picks 8 of 256 experts.
print("\ntokens per expert per step (even routing), 128 sequences per GPU:")
for ep in (8, 16, 72, 144):
    tokens = ep * 128 * active / routed
    print(f"  EP{ep:<4}{tokens:7.0f} tokens per expert")
