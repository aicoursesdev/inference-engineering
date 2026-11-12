"""Price Tidewater's average request, input and output apart, from measured speeds.

Speeds: this laptop, Qwen3-4B-Q4_K_M. Prefill 907 tokens/s (llama-bench pp512, see llama-bench-4b-q4km.json),
decode 284 tokens/s aggregate at 16 requests in flight (01-price-of-a-token/throughput_vs_load.py).
The 1.00 dollars an hour is a round number for easy checking, not a price.
"""
from servinglab.cost import cost_per_mtok, request_cost

r = request_cost(rate=1.00, prefill_tps=907, decode_tps=284, input_tokens=403, output_tokens=144)
print(f"GPU time: {r['gpu_seconds']:.3f} s per request, cost ${r['usd']:.6f}")
print(f"input  ${r['input_per_mtok']:.3f} per million tokens")
print(f"output ${r['output_per_mtok']:.3f} per million tokens")
print(f"output only, ignoring input: ${cost_per_mtok(1.00, 284):.3f}")
# published node-level figures, with a GPU price from our price sheet
for label, node_rate, out_tps in [("LMSYS 8xH100, RunPod community 2.69/GPU-h", 8 * 2.69, 22300),
                                  ("LMSYS 8xH100, Modal 3.9492/GPU-h", 8 * 3.9492, 22300)]:
    print(f"{label}: ${cost_per_mtok(node_rate, out_tps):.3f} per M output tokens")
print(f"rate that makes LMSYS's 0.20 true: ${0.20 * 22300 * 3600 / 1e6 / 8:.2f} per GPU-hour")
print(f"DeepSeek all-in: ${87072 / 168e3:.3f} per M output tokens (87,072 dollars / 168 billion)")
