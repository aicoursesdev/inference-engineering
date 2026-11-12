"""Static against continuous batching on Monsoon's nightly batch (simulated).

    uv run python nightly_batch.py

800 summarising requests arrive within two minutes. Simulated Qwen3-4B in 16-bit
on an L4: Module 02's roofline prices each step, so these are predictions.
"""

from servinglab.data import traces
from scheduler import Scheduler, StepModel, report, simulate, window

rows = window(traces("mixed"), 2, 3, ["batch"])
print(f"{len(rows)} requests\n")
print(f"{'policy':11s} {'batch':>5s} {'output tok/s':>13s} {'finished at':>12s} {'mean batch':>11s}")
for name, kw in (("static", dict(mode="static")),
                 ("continuous", dict(chunked=False, token_budget=10**9))):
    for slots in (4, 16):
        sched = Scheduler(num_blocks=5000, max_seqs=slots, **kw)
        reqs, steps = simulate(rows, sched, StepModel("Qwen3-4B", "L4"))
        r = report(reqs, steps)
        end = max(q.stamps[-1] for q in reqs) - min(q.arrival for q in reqs)
        print(f"{name:11s} {slots:5d} {r['tokens_per_s']:13.0f} {end:10.0f} s {r['mean_batch']:11.1f}")
