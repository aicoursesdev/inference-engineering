"""Static, continuous and chunked scheduling across load, in the simulator.

    uv run python compare_policies.py [--plot curves.png]

Workload: Monsoon's chat and contract traffic from 10:00 to 12:00,
replayed faster to raise the load. Simulated Qwen3-4B in 16-bit on an L4.
"""

import sys

from servinglab.data import traces
from scheduler import Scheduler, StepModel, report, simulate, window

rows = window(traces("mixed"), 10, 12, ["interactive", "long"])
POLICIES = {"static": dict(mode="static"),
            "continuous": dict(chunked=False, token_budget=10**9),
            "chunked": dict(chunked=True, token_budget=512)}
SPEEDUPS = (5, 10, 20, 40)
duration = rows[-1]["arrival_s"] - rows[0]["arrival_s"]
print(f"{len(rows)} requests over {duration / 3600:.1f} h\n")
print(f"{'speed-up':>8s} {'req/s':>6s} {'policy':11s} {'tok/s':>6s} {'goodput':>8s} "
      f"{'TTFT p95':>9s} {'gap p99':>8s} {'met target':>10s}")
curves = {p: [] for p in POLICIES}
for speed in SPEEDUPS:
    for name, kw in POLICIES.items():
        sched = Scheduler(num_blocks=5000, max_seqs=32, **kw)
        reqs, steps = simulate(rows, sched, StepModel("Qwen3-4B", "L4"), scale=speed)
        r = report(reqs, steps)
        offered = len(rows) / (duration / speed)
        curves[name].append((offered, r))
        print(f"{speed:8d} {offered:6.2f} {name:11s} {r['tokens_per_s']:6.0f} "
              f"{r['goodput_rps']:8.2f} {r['ttft_p95']:8.1f}s {r['itl_p99'] * 1000:5.0f} ms "
              f"{r['attainment']:10.0%}")

if "--plot" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, pts in curves.items():
        axes[0].plot([o for o, _ in pts], [r["goodput_rps"] for _, r in pts], "o-", label=name)
        axes[1].plot([o for o, _ in pts], [r["itl_p99"] * 1000 for _, r in pts], "o-", label=name)
    axes[0].set(xlabel="offered requests/s", ylabel="goodput (requests/s)")
    axes[1].set(xlabel="offered requests/s", ylabel="p99 gap between tokens (ms)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(sys.argv[sys.argv.index("--plot") + 1], dpi=130)
