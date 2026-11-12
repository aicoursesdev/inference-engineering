"""Static versus continuous batching on six requests, one unit of time a step.

    uv run python step_loop.py
"""

from scheduler import Request, Scheduler

LENGTHS = [20, 60, 30, 200, 40, 10]          # tokens each request will write
SLOTS = 3                                     # the batch holds three requests


def run(mode):
    sched = Scheduler(num_blocks=512, max_seqs=SLOTS, mode=mode, token_budget=4096)
    reqs = [Request(f"r{i}", [10 * i + j for j in range(8)], n) for i, n in enumerate(LENGTHS)]
    for r in reqs:
        sched.submit(r)
    steps = slots_used = 0
    started, ended = {}, {}
    while not sched.idle:
        plan = sched.schedule()
        steps += 1
        slots_used += sum(1 for r, n in plan if n > 0)       # rows doing real work
        for r, n in plan:
            started.setdefault(r.id, steps)
        sched.update(plan, float(steps))
        for r in reqs:
            if r.status == "done":
                ended.setdefault(r.id, steps)
    waits = [started[r.id] - 1 for r in reqs]
    print(f"{mode:10s} {steps:4d} steps, slots busy {slots_used / (steps * SLOTS):4.0%}, "
          f"waits before starting {waits}")
    return steps


a, b = run("static"), run("continuous")
print(f"continuous finishes the same work in {b / a:.0%} of the steps")
