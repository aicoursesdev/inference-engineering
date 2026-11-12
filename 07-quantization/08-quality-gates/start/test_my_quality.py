"""Your functions against the finished servinglab.quality, and the lesson's own cases."""
import numpy as np

import my_quality as mine
from servinglab import quality as done

RNG = np.random.default_rng(0)
BASE, TEST = RNG.normal(size=(30, 50)), RNG.normal(size=(30, 50))
TARGETS = RNG.integers(0, 50, size=30)


def test_a_uniform_model_has_perplexity_equal_to_the_vocabulary_size():
    assert abs(mine.perplexity(np.zeros((5, 10)), np.arange(5)) - 10) < 1e-9


def test_perplexity_matches_the_finished_one():
    assert abs(mine.perplexity(BASE, TARGETS) - done.perplexity(BASE, TARGETS)) < 1e-9


def test_kl_is_zero_for_the_same_logits_and_positive_otherwise():
    assert mine.kl_divergence(BASE, BASE) == 0
    assert mine.kl_divergence(BASE, TEST) > 0
    assert abs(mine.kl_divergence(BASE, TEST) - done.kl_divergence(BASE, TEST)) < 1e-12


def test_top1_agreement_counts_matching_argmax_positions():
    assert mine.top1_agreement(BASE, BASE) == 1.0
    assert mine.top1_agreement(np.eye(4), np.eye(4)[::-1]) == 0.0
    assert mine.top1_agreement(BASE, TEST) == done.top1_agreement(BASE, TEST)


def test_accuracy_is_the_share_of_answers_equal_to_gold():
    assert mine.accuracy(["a", "b", "c", "d"], ["a", "x", "c", "y"]) == 0.5


def test_the_gate_passes_a_close_build_and_fails_each_rule_alone():
    base = {"ppl": 20.0, "task": 0.8}
    ok = {"ppl": 20.5, "task": 0.8, "kl": 0.01, "top1": 0.95}
    assert mine.gate(base, ok) == (True, [])
    for broken in ({"ppl": 22}, {"kl": 0.2}, {"top1": 0.8}, {"task": 0.7}):
        passed, why = mine.gate(base, {**ok, **broken})
        assert not passed and len(why) == 1


def test_the_gate_names_every_reason_and_accepts_your_own_limits():
    base = {"ppl": 20.0, "task": 0.8}
    bad = {"ppl": 40.0, "task": 0.1, "kl": 2.0, "top1": 0.3}
    assert len(mine.gate(base, bad)[1]) == 4
    assert mine.gate(base, bad, {"ppl_ratio_max": 3, "kl_max": 3, "top1_min": 0, "task_drop_max": 1})[0]


def test_the_reasons_read_as_in_the_lessons_table():
    ok, why = mine.gate({"ppl": 20.671296, "task": 0.675},
                        {"ppl": 22.415193, "task": 0.75, "kl": 0.104539, "top1": 0.83592})
    assert not ok
    assert why == ["perplexity 1.084x the base", "KL 0.1045 over 0.05",
                   "top-1 agreement 0.836 under 0.9"]
