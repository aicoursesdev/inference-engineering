"""Little's law on a finished run: L = lambda x W, counted two ways."""

import numpy as np


def littles_law(done, lam):
    """`done`: finished requests (each has `arrival` and `stamps`, the times
    its tokens were made); `lam`: arrivals per second. Returns (W, lam x W, L):
    W is the mean seconds from arrival to the last token, lam x W the number
    in the system Little's law predicts, and L the number counted directly,
    the total request-seconds spent inside divided by the length of the run."""
    inside = [r.stamps[-1] - r.arrival for r in done]
    span = max(r.stamps[-1] for r in done) - min(r.arrival for r in done)
    w = float(np.mean(inside))
    return w, lam * w, sum(inside) / span
