"""One long contract arrives among chat users: how long does everyone's stream freeze?

    uv run python chunk_sweep.py

Simulated: Qwen3-4B in 16-bit on an L4 (Module 02's roofline prices each step),
Tidewater's festival-sale half hour (12:00 to 12:30) plus Nyaya's contracts.
"""

from servinglab.data import traces
from scheduler import Scheduler, StepModel, report, simulate, window

rows = window(traces("mixed"), 12, 12.5, ["interactive", "long"])
longs = sum(r["slo_class"] == "long" for r in rows)
print(f"{len(rows)} requests, {longs} of them contracts of 20,000 to 58,000 tokens (median TTFT shown)\n")
print(f"{'policy':22s} {'worst freeze':>12s} {'p99 gap':>8s} {'chat TTFT p95':>14s} {'contract TTFT':>14s} {'steps':>6s}")
for name, kw in (("whole prompt in a step", dict(chunked=False, token_budget=10**9)),
                 ("chunked, budget 2048", dict(token_budget=2048)),
                 ("chunked, budget 512", dict(token_budget=512)),
                 ("chunked, budget 256", dict(token_budget=256))):
    sched = Scheduler(num_blocks=5000, max_seqs=64, **kw)
    reqs, steps = simulate(rows, sched, StepModel("Qwen3-4B", "L4"))
    r = report(reqs, steps)
    chat = sorted(q.stamps[0] - q.arrival for q in reqs if q.slo == "interactive")
    contract = sorted(q.stamps[0] - q.arrival for q in reqs if q.slo == "long")
    print(f"{name:22s} {r['itl_max'] * 1000:9.0f} ms {r['itl_p99'] * 1000:5.0f} ms "
          f"{chat[int(0.95 * len(chat))]:13.2f}s {contract[len(contract) // 2]:13.2f}s {len(steps):6d}")
