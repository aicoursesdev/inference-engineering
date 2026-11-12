"""Offline checks for the project: greedy speculation equals greedy decoding,
and a perfect draft is always accepted. Uses the tiny test model."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from servinglab.engine import Config, Model
from servinglab.engine.weights import read_safetensors
from servinglab.kv import generate_cached
from speculate import (DraftModel, NgramProposer, accept_reject, speculate,
                       verify)

TINY = Path(__file__).resolve().parents[3] / "servinglab/tests/data/tiny-llama"


def tiny(noise=0.0):
    c = Config.from_json(json.loads((TINY / "config.json").read_text()))
    c = Config(**{**c.__dict__, "eos": (-1,)})
    rng = np.random.default_rng(1)
    w = {k: v + noise * rng.standard_normal(v.shape).astype(np.float32)
         for k, v in read_safetensors(TINY / "model.safetensors").items()}
    return Model(c, w, None, TINY)


PROMPT = [5, 17, 3, 99, 42, 7, 7, 61]


def test_greedy_speculation_is_greedy_decoding():
    want = list(generate_cached(tiny(), PROMPT, 24, temperature=0))
    for proposer in (DraftModel(tiny(noise=0.05)), NgramProposer()):
        got = speculate(tiny(), proposer, [PROMPT], k=4, max_new_tokens=24)
        assert got.tokens[0] == want


def test_a_perfect_draft_is_always_accepted():
    out = speculate(tiny(), DraftModel(tiny()), [PROMPT], k=3, max_new_tokens=24)
    assert out.stats.acceptance_rate == 1.0


def test_the_rule_gives_the_target_distribution_even_from_a_bad_guesser():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    q = np.array([0.1, 0.1, 0.1, 0.7])
    rng, seen = np.random.default_rng(0), Counter()
    for _ in range(40000):
        seen[accept_reject(p, q, int(rng.choice(4, p=q)), rng)[1]] += 1
    freq = np.array([seen[i] for i in range(4)]) / 40000
    assert np.abs(freq - p).max() < 0.01


def test_verify_returns_one_to_k_plus_one_tokens():
    p = np.eye(3)[[0, 1, 2, 0]]       # a target that is certain: 0, 1, 2, 0
    rng = np.random.default_rng(0)
    assert verify(p, [0, 1, 2], [None] * 3, rng) == [0, 1, 2, 0]   # all kept
    assert verify(p, [0, 2, 2], [None] * 3, rng) == [0, 1]         # one miss


def test_the_ngram_speculator_copies_what_followed_the_last_match():
    guesses, qs = NgramProposer().propose([[1, 2, 3, 9, 8, 1, 2, 3]], [0], 2,
                                          0.0, {}, None)
    assert guesses == [[9, 8]] and qs == [[None, None]]
    assert NgramProposer().propose([[1, 2, 3, 4]], [0], 2, 0.0, {}, None)[0] == [[]]


def test_the_draft_model_gives_k_guesses_and_their_probabilities():
    draft = DraftModel(tiny())
    draft.start(1)
    guesses, qs = draft.propose([PROMPT], [0], 3, 0.0, {},
                                np.random.default_rng(0))
    assert len(guesses[0]) == 3 and len(qs[0]) == 3
    assert all(abs(q.sum() - 1) < 1e-4 for q in qs[0])
