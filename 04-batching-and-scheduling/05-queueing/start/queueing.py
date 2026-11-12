"""Little's law on a finished run: L = lambda x W, counted two ways."""

import numpy as np


def littles_law(done, lam):
    """`done`: finished requests (each has `arrival` and `stamps`, the times
    its tokens were made); `lam`: arrivals per second. Returns (W, lam x W, L):
    W is the mean seconds from arrival to the last token, lam x W the number
    in the system Little's law predicts, and L the number counted directly,
    the total request-seconds spent inside divided by the length of the run."""
    raise NotImplementedError(
        "TODO 04.5: return W, lam * W and the request-seconds spent inside "
        "divided by the run's length, see the lesson's "
        "'Little’s law: counting people in a shop' section")
