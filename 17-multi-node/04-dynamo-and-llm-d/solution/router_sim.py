"""Round-robin vs cache-aware routing for Kodekart-style agent sessions.

Each session sends 12 steps. Step k's prompt is the session's own history
(6,000 tokens plus 800 per step), all of it already seen by the worker that
served step k-1. A worker that saw the session before skips that prefill.
shortcut: caches never evict; add an LRU limit to see capacity effects.
"""
import random

WORKERS, SESSIONS, STEPS = 8, 40, 12
BASE, GROW = 6000, 800


def run(policy, seed=7):
    rng = random.Random(seed)
    # interleave steps from many sessions, as real traffic does
    arrivals = []
    for _ in range(STEPS):
        order = list(range(SESSIONS))
        rng.shuffle(order)
        arrivals += order
    cached = [dict() for _ in range(WORKERS)]    # worker -> {session: tokens}
    load = [0] * WORKERS
    step_of = [0] * SESSIONS
    hit = total = 0
    for i, s in enumerate(arrivals):
        prompt = BASE + GROW * step_of[s]
        step_of[s] += 1
        if policy == "round-robin":
            w = i % WORKERS
        else:   # cache-aware: the worker with the longest cached prefix,
            # unless it is already 30% busier than the average (then least loaded)
            best = max(range(WORKERS), key=lambda x: cached[x].get(s, 0))
            avg = sum(load) / WORKERS
            w = best if load[best] <= 1.3 * avg + 1 else load.index(min(load))
        hit += min(cached[w].get(s, 0), prompt)
        total += prompt
        cached[w][s] = prompt
        load[w] += 1
    return hit / total, max(load) / (sum(load) / WORKERS)


for p in ["round-robin", "cache-aware"]:
    frac, skew = run(p)
    print(f"{p:12} prefill skipped: {frac:5.1%}   busiest worker: {skew:.2f}x average")
