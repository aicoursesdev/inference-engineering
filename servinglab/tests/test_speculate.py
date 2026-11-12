"""Speculative decoding (Module 08). Offline: a tiny model, no network."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from servinglab.engine import Config, Model
from servinglab.engine.weights import read_safetensors
from servinglab.kv import generate_cached
from servinglab.speculate import (DraftModel, NgramProposer, accept_reject,
                                  expected_speedup, expected_tokens, speculate,
                                  verify)

DATA = Path(__file__).parent / "data" / "tiny-llama"


def tiny(noise=0.0, seed=1):
    """The tiny model, with `noise` added to every weight to make a worse copy."""
    config = Config.from_json(json.loads((DATA / "config.json").read_text()))
    config = Config(**{**config.__dict__, "eos": (-1,)})  # never stop early
    w = read_safetensors(DATA / "model.safetensors")
    rng = np.random.default_rng(seed)
    w = {k: v + noise * rng.standard_normal(v.shape).astype(np.float32)
         for k, v in w.items()}
    return Model(config, w, None, DATA)


PROMPT = [5, 17, 3, 99, 42, 7, 7, 61]


def test_the_rule_keeps_the_target_distribution():
    # a guesser that is wrong about everything still gives samples from p
    p = np.array([0.5, 0.3, 0.15, 0.05])
    q = np.array([0.1, 0.1, 0.1, 0.7])
    rng, seen = np.random.default_rng(0), Counter()
    for _ in range(40000):
        x = int(rng.choice(4, p=q))
        seen[accept_reject(p, q, x, rng)[1]] += 1
    freq = np.array([seen[i] for i in range(4)]) / 40000
    assert np.abs(freq - p).max() < 0.01


def test_point_mass_guesses_keep_the_distribution_too():
    p = np.array([0.5, 0.3, 0.2])
    rng, seen = np.random.default_rng(1), Counter()
    for _ in range(30000):
        seen[verify(np.stack([p, p]), [1], [None], rng)[0]] += 1
    assert abs(seen[1] / 30000 - 0.3) < 0.01   # guess 1, kept with p[1]
    assert abs(seen[0] / 30000 - 0.5) < 0.01


def test_verify_returns_one_to_k_plus_one_tokens():
    p = np.eye(3)[[0, 1, 2, 0]]   # a target that is certain: 0, 1, 2, then 0
    rng = np.random.default_rng(0)
    assert verify(p, [0, 1, 2], [None] * 3, rng) == [0, 1, 2, 0]   # all kept
    assert verify(p, [0, 2, 2], [None] * 3, rng) == [0, 1]         # one miss


def test_greedy_speculation_equals_greedy_decoding():
    target = tiny()
    want = list(generate_cached(target, PROMPT, 30, temperature=0))
    for proposer in (DraftModel(tiny(noise=0.05)), NgramProposer()):
        out = speculate(target, proposer, [PROMPT], k=4, max_new_tokens=30)
        assert out.tokens[0] == want


def test_a_perfect_draft_is_always_accepted():
    target = tiny()
    out = speculate(target, DraftModel(tiny()), [PROMPT], k=3,
                    max_new_tokens=24)
    assert out.stats.acceptance_rate == 1.0
    assert out.stats.tokens_per_step > 3   # up to k + 1 = 4 a step


def test_a_batch_gives_the_same_tokens_as_one_at_a_time():
    target, prompts = tiny(), [PROMPT, [9, 8, 7], [1, 2, 3, 4, 5]]
    alone = [speculate(target, DraftModel(tiny(noise=0.05)), [p], k=3,
                       max_new_tokens=16).tokens[0] for p in prompts]
    both = speculate(target, DraftModel(tiny(noise=0.05)), prompts, k=3,
                     max_new_tokens=16).tokens
    assert both == alone


def test_sampled_speculation_runs_and_counts_tokens():
    out = speculate(tiny(), DraftModel(tiny(noise=0.1)), [PROMPT], k=3,
                    max_new_tokens=20, temperature=0.8, top_p=0.95)
    assert len(out.tokens[0]) == 20
    assert out.stats.tokens == 20 and out.stats.steps >= 20 / 4


def test_ngram_copies_from_the_text():
    guess, _ = NgramProposer(max_n=2)._one([4, 5, 6, 7, 8, 1, 4, 5], 3)
    assert guess == [6, 7, 8]
    assert NgramProposer()._one([1, 2, 3], 3) == ([], [])


def test_the_formulas():
    assert expected_tokens(0.0, 4) == 1
    assert expected_tokens(1.0, 4) == 5
    assert abs(expected_tokens(0.8, 4) - (1 - 0.8 ** 5) / 0.2) < 1e-12
    assert abs(expected_speedup(0.8, 4, 0.0) - expected_tokens(0.8, 4)) < 1e-12
    assert expected_speedup(0.5, 4, 0.5) < 1   # a draft this slow loses
