"""Profile Monsoon's four traces: how long are prompts and replies, how much of
each prompt could a prefix cache reuse, and how bursty is the arrival rate?

    uv run python traffic_shapes.py
"""

import numpy as np

from servinglab.data import traces
from servinglab.replay import build_requests, trace_hit_rate

print(f"{'trace':10} {'requests':>8} {'median in':>10} {'median out':>10} "
      f"{'in:out':>7} {'ceiling':>9} {'peak/mean rate':>15}")
for name in ("tidewater", "kodekart", "nyaya", "batch"):
    rows = traces(name)
    n_in = np.array([r["input_tokens"] for r in rows])
    n_out = np.array([r["output_tokens"] for r in rows])
    reuse = trace_hit_rate(build_requests(name))
    t = np.array([r["arrival_s"] for r in rows])
    # requests per minute, over the minutes the trace is active
    per_min = np.bincount((t // 60).astype(int))
    per_min = per_min[per_min.nonzero()[0].min():]
    peak = per_min.max() / per_min.mean()
    print(f"{name:10} {len(rows):8d} {np.median(n_in):10.0f} "
          f"{np.median(n_out):10.0f} {n_in.sum() / n_out.sum():7.1f} "
          f"{reuse:9.0%} {peak:15.1f}")
