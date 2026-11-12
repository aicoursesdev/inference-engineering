"""Little's law and the saturation knee, on Tidewater's chat traffic.

    uv run python knee.py

Simulated Qwen3-4B on an L4. Chat arrivals are replayed faster to raise the
load; capacity is what the same server finishes when everything arrives at once.
"""

import numpy as np

from servinglab.data import traces
from queueing import littles_law
from scheduler import Scheduler, StepModel, simulate, timed, window

chat = window(traces("mixed"), 9, 18, ["interactive"])


def run(scale):
    reqs, steps = simulate(chat, Scheduler(num_blocks=5000, max_seqs=64), StepModel(), scale=scale)
    done = [r for r in reqs if r.status == "done"]
    span = max(r.stamps[-1] for r in done) - min(r.arrival for r in done)
    return reqs, done, span


# Capacity: every request arrives together, so the server is never idle.
_, done, span = run(1e9)
capacity = len(done) / span
print(f"capacity: {len(done)} requests in {span:.1f} s = {capacity:.1f} requests/s\n")

print(f"{'load':>5s} {'arrivals/s':>10s} {'W mean':>7s} {'L = arr x W':>11s} {'L counted':>9s} "
      f"{'TTFT p95':>9s}")
for load in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.1):
    # pick the speed-up that offers `load` x capacity over the trace's own length
    scale = load * capacity * (chat[-1]["arrival_s"] - chat[0]["arrival_s"]) / len(chat)
    reqs, done, span = run(scale)
    lam = len(done) / (max(r.arrival for r in done) - min(r.arrival for r in done))
    w, predicted, counted = littles_law(done, lam)
    ttft = np.percentile([t.ttft for t in timed(reqs)], 95)
    print(f"{load:5.2f} {lam:10.2f} {w:6.2f}s {predicted:11.1f} {counted:9.1f} {ttft:8.2f}s")
