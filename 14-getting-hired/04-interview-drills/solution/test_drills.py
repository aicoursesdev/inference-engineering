"""Offline checks for the three drills (no GPU, no network)."""
import numpy as np

import capacity_drill
import scheduler_drill
import stable_attention
from scheduler_drill import Req


def test_capacity_matches_the_lesson():
    at_ceiling = capacity_drill.gpus_at(1.0)
    assert round(at_ceiling["busy"], 2) == 1.54
    assert at_ceiling["gpus"] == 4
    assert capacity_drill.gpus_at(0.5)["gpus"] == 6


def plans(budget=512, max_seqs=4):
    running = [Req("a", 0, True), Req("b", 0, True)]
    waiting = [Req("c", 900), Req("d", 50)]
    return [scheduler_drill.step(running, waiting, budget, max_seqs)
            for _ in range(3)]


def test_scheduler_matches_the_lesson():
    assert plans() == [
        {"a": 1, "b": 1, "c": 510},
        {"a": 1, "b": 1, "c": 390, "d": 50},
        {"a": 1, "b": 1, "c": 1, "d": 1},
    ]


def test_scheduler_never_exceeds_the_budget():
    for plan in plans(budget=100):
        assert sum(plan.values()) <= 100


def test_softmax_survives_big_scores():
    big = np.array([1000.0, 1001.0, 1002.0])
    out = stable_attention.softmax(big)
    assert np.allclose(out, [0.09003057, 0.24472847, 0.66524096])
    assert np.isclose(out.sum(), 1.0)


def test_causal_attention_only_looks_back():
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((5, 4)) for _ in range(3))
    out = stable_attention.causal_attention(q, k, v)
    assert np.allclose(out[0], v[0])          # token 0 sees only itself
    v2 = v.copy()
    v2[4] += 10                               # a later token changes...
    assert np.allclose(stable_attention.causal_attention(q, k, v2)[:4], out[:4])
