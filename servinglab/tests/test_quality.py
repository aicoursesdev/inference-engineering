import numpy as np

from servinglab import quality as q


def test_round_trip_error_shrinks_with_bits():
    w = np.random.default_rng(0).normal(size=(64, 64))
    e8, e4 = (q.rms_error(w, q.round_to_bits(w, b)) for b in (8, 4))
    assert e8 < e4 / 10


def test_groups_beat_one_scale_when_an_outlier_exists():
    w = np.random.default_rng(1).normal(size=4096)
    w[0] = 40.0
    assert q.rms_error(w, q.round_to_bits(w, 4, 32)) < \
        q.rms_error(w, q.round_to_bits(w, 4))


def test_perplexity_of_uniform_is_vocab_size():
    assert abs(q.perplexity(np.zeros((5, 10)), np.arange(5)) - 10) < 1e-9


def test_kl_zero_for_same_and_positive_for_different():
    a = np.random.default_rng(2).normal(size=(6, 20))
    assert q.kl_divergence(a, a) == 0
    assert q.kl_divergence(a, a + np.random.default_rng(3).normal(size=a.shape)) > 0


def test_gate_fails_on_each_rule():
    base = {"ppl": 20.0, "task": 0.8}
    ok = {"ppl": 20.5, "task": 0.8, "kl": 0.01, "top1": 0.95}
    assert q.gate(base, ok) == (True, [])
    assert not q.gate(base, {**ok, "ppl": 22})[0]
    assert not q.gate(base, {**ok, "kl": 0.2})[0]
    assert not q.gate(base, {**ok, "top1": 0.8})[0]
    assert not q.gate(base, {**ok, "task": 0.7})[0]
