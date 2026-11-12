import numpy as np

from servinglab.data import traces
from queueing import littles_law
from scheduler import Scheduler, StepModel, simulate, timed, window

CHAT = window(traces("mixed"), 9, 18, ["interactive"])[:300]


def ttft_p95(scale):
    reqs, _ = simulate(CHAT, Scheduler(num_blocks=5000), StepModel(), scale=scale)
    return np.percentile([t.ttft for t in timed(reqs)], 95)


def test_waiting_time_climbs_as_load_rises():
    assert ttft_p95(200) < ttft_p95(600) < ttft_p95(1500)


def test_littles_law_holds_for_a_finished_run():
    reqs, _ = simulate(CHAT, Scheduler(num_blocks=5000), StepModel(), scale=400)
    span = max(r.stamps[-1] for r in reqs) - min(r.arrival for r in reqs)
    w, predicted, counted = littles_law(reqs, len(reqs) / span)
    assert w > 0 and abs(predicted - counted) < 1e-6


def test_littles_law_by_hand():
    class R:
        def __init__(self, arrival, end):
            self.arrival, self.stamps = arrival, [end]
    # two requests in a 6 s run, inside for 4 s each: L = 8 / 6, lambda = 2 / 6
    w, predicted, counted = littles_law([R(0, 4), R(2, 6)], 2 / 6)
    assert (w, round(predicted, 6), round(counted, 6)) == (4.0, 1.333333, 1.333333)
