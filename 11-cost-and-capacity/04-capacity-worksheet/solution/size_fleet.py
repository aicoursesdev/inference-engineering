"""Size a fleet for each Monsoon customer's busiest minute, from its trace and two measured speeds.

Speeds are this laptop's, measured with Qwen3-4B-Q4_K_M: prefill 907 tokens/s
(llama-bench pp512) and decode 284 tokens/s in aggregate at 16 requests in flight
(01-price-of-a-token/throughput_vs_load.py). Swap in your own numbers.
The tokens are counted as if no prefix were cached, the worst case.
"""
from collections import Counter

from servinglab.cost import gpus_needed
from servinglab.data import traces

PREFILL_TPS, DECODE_TPS = 907, 284

print("customer   peak minute  QPS   mean in/out   busy machines  with 30% headroom + 1 spare")
for name in ("tidewater", "kodekart", "nyaya"):
    rows = traces(name)
    per_minute = Counter(int(r["arrival_s"] // 60) for r in rows)
    minute, n = per_minute.most_common(1)[0]
    mean_in = sum(r["input_tokens"] for r in rows) / len(rows)
    mean_out = sum(r["output_tokens"] for r in rows) / len(rows)
    g = gpus_needed(n / 60, mean_in, mean_out, PREFILL_TPS, DECODE_TPS, headroom=0.3, spares=1)
    print(f"{name:<10} {minute // 60:02d}:{minute % 60:02d}  {n / 60:7.2f}  {mean_in:6.0f}/{mean_out:<4.0f}"
          f"  {g['busy']:13.2f}  {g['gpus']:>5}")

# The batch has no one waiting: spread its work over its 6-hour deadline instead of its 2-minute burst.
rows = traces("batch")
work = sum(r["input_tokens"] / PREFILL_TPS + r["output_tokens"] / DECODE_TPS for r in rows)
print(f"batch: {work:.0f} machine-seconds of work; over 6 h that is {work / (6 * 3600):.2f} machines busy")
