"""Memory runs out, then the queue grows: what the scheduler does about it.

    uv run python pressure.py

Simulated Qwen3-4B on an L4. Part 1 shrinks the cache pool; part 2 caps the
queue; part 3 lets chat overtake the nightly batch.
"""

from servinglab.data import traces
from scheduler import Scheduler, StepModel, report, simulate, window

mixed = traces("mixed")
model = StepModel("Qwen3-4B", "L4")


def run(rows, scale=1, **kw):
    sched = Scheduler(**kw)
    reqs, steps = simulate(rows, sched, model, scale=scale)
    return report(reqs, steps), reqs


# 1. Preemption: the nightly batch (800 requests in two minutes) in a shrinking pool.
batch = window(mixed, 2, 3, ["batch"])
print(f"{'pool (blocks)':>13s} {'preemptions':>11s} {'finished':>8s} {'tok/s':>6s}")
for blocks in (5000, 2500, 1200, 600):
    r, _ = run(batch, num_blocks=blocks, max_seqs=64)
    print(f"{blocks:13d} {r['preemptions']:11d} {r['finished']:8d} {r['tokens_per_s']:6.0f}")

# 2. Admission control: Tidewater replayed 300x faster (a flood), with and without a queue cap.
chat = window(mixed, 9, 18, ["interactive"])
print(f"\n{'queue cap':>9s} {'turned away':>11s} {'TTFT p95':>9s} {'met their target':>17s}")
for cap in (None, 200, 50, 10):
    r, _ = run(chat, scale=300, num_blocks=5000, max_seqs=64, max_queue=cap)
    print(f"{str(cap):>9s} {r['rejected']:11d} {r['ttft_p95']:8.1f}s {r['attainment']:17.0%}")

# 3. Priorities: batch jobs and chat together in the 02:00 burst, chat arriving in the middle.
both = window(mixed, 2, 3, ["batch"]) + [{**r, "arrival_s": r["arrival_s"] % 3600 / 40 + 30}
                                       for r in window(mixed, 12, 12.5, ["interactive"])[:60]]
print(f"\n{'policy':9s} {'chat TTFT p95':>14s}")
for policy in ("fcfs", "priority"):
    _, reqs = run(both, num_blocks=5000, max_seqs=16, policy=policy)
    chat_ttft = sorted(r.stamps[0] - r.arrival for r in reqs if r.slo == "interactive" and r.stamps)
    print(f"{policy:9s} {chat_ttft[int(0.95 * len(chat_ttft))]:13.1f}s")
