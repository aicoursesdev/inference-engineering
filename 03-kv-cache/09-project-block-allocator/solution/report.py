"""The project's report: three allocation policies on Monsoon's traces.

    uv run python report.py

Each trace is replayed faster than real time (SCALE) so requests overlap and
memory runs short, into a pool of BLOCKS blocks of 16 tokens.
"""

from servinglab.data import traces
from servinglab.kv import replay

# trace, requests, replay speed-up, blocks in the pool
RUNS = [("tidewater", 800, 100, 600), ("kodekart", 641, 100, 12000), ("nyaya", 100, 400, 14000)]

print(f"{'trace':10s} {'policy':8s} {'admitted':>8s} {'turned away':>11s} "
      f"{'waste':>6s} {'cache hits':>10s} {'peak tokens':>11s}")
for name, n, scale, blocks in RUNS:
    rows = [{**r, "arrival_s": r["arrival_s"] / scale} for r in traces(name)[:n]]
    for mode in ("reserve", "paged", "shared"):
        r = replay(rows, blocks, mode)
        print(f"{name:10s} {mode:8s} {r['admitted']:8d} {r['rejected']:11d} "
              f"{r['mean_waste']:6.1%} {r['hit_rate']:10.0%} {r['peak_slots']:11,d}")
