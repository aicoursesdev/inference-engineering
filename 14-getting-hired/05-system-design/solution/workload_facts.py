"""Step 1 of every serving design: say what the traffic is, from a trace, before any GPU.

For each Monsoon customer: requests, busiest minute, token percentiles and the
share of prompt tokens that a prefix cache could reuse (prefix_tokens / input_tokens)."""
from collections import Counter

import numpy as np

from servinglab.data import traces

print(f"{'customer':<10} {'reqs':>5} {'peak req/s':>10} {'in p50/p95':>12} "
      f"{'out p50/p95':>12} {'shared prefix':>14}")
for name in ("tidewater", "kodekart", "nyaya", "batch"):
    rows = traces(name)
    per_min = Counter(int(r["arrival_s"] // 60) for r in rows)
    peak = max(per_min.values()) / 60
    i = np.array([r["input_tokens"] for r in rows])
    o = np.array([r["output_tokens"] for r in rows])
    share = sum(r["prefix_tokens"] for r in rows) / i.sum()
    print(f"{name:<10} {len(rows):>5} {peak:>10.2f} "
          f"{np.percentile(i, 50):>5.0f}/{np.percentile(i, 95):<6.0f}"
          f"{np.percentile(o, 50):>5.0f}/{np.percentile(o, 95):<6.0f}{share:>14.0%}")
