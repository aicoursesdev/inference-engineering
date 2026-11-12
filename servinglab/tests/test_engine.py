"""The engine against transformers' own outputs (saved by
scripts/make_engine_reference.py), plus the sampling rules.

The tiny-model tests run anywhere. The SmolLM2 tests need the real model in
models/ and are skipped when it is not there.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from servinglab.engine import (Config, Model, apply_penalties, generate,
                               probabilities, sample)
from servinglab.engine.model import MODELS
from servinglab.engine.weights import bf16_to_f32, read_safetensors

DATA = Path(__file__).parent / "data"
SMOL = MODELS / "SmolLM2-135M-Instruct"


def tiny_model():
    d = DATA / "tiny-llama"
    config = Config.from_json(json.loads((d / "config.json").read_text()))
    return Model(config, read_safetensors(d / "model.safetensors"), None, d)


def test_tiny_logits_match_transformers():
    ref = json.loads((DATA / "tiny-llama" / "reference.json").read_text())
    ours = tiny_model().forward(ref["ids"])
    assert ours.shape == (len(ref["ids"]), 100)
    assert np.abs(ours - np.array(ref["logits"])).max() < 1e-3


def test_last_only_is_the_last_row():
    m, ids = tiny_model(), [5, 17, 3, 99]
    assert np.allclose(m.forward(ids, last_only=True)[0], m.forward(ids)[-1])


def test_causal_mask_later_tokens_do_not_change_earlier_logits():
    m = tiny_model()
    a = m.forward([5, 17, 3, 99, 42])[:3]
    b = m.forward([5, 17, 3])
    assert np.allclose(a, b, atol=1e-5)


def test_param_count_formula_matches_the_real_tensors():
    m = tiny_model()
    assert m.config.n_params() == m.n_params


def test_generate_greedy_is_deterministic_and_stops_at_the_limit():
    m = tiny_model()
    m.config = Config(**{**m.config.__dict__, "eos": (-1,)})
    a = list(generate(m, [5, 17], max_new_tokens=6, temperature=0))
    b = list(generate(m, [5, 17], max_new_tokens=6, temperature=0))
    assert a == b and len(a) == 6


def test_bf16_widening():
    # 1.0 is 0x3F80 in bfloat16 and -2.0 is 0xC000
    out = bf16_to_f32(np.array([0x3F80, 0xC000], dtype=np.uint16))
    assert out.tolist() == [1.0, -2.0]


@pytest.mark.skipif(not (SMOL / "model.safetensors").is_file(),
                    reason="run load('SmolLM2-135M-Instruct') once first")
def test_smollm2_matches_transformers():
    from servinglab.engine import load
    ref = np.load(DATA / "smollm2-135m.npz")
    m = load("SmolLM2-135M-Instruct")
    assert m.n_params == m.config.n_params() == 134_515_008
    logits = m.forward(ref["prompt_ids"], last_only=True)[0]
    assert np.abs(logits - ref["last_logits"]).max() < 1e-3
    assert int(np.argmax(logits)) == int(np.argmax(ref["last_logits"]))
    greedy = list(generate(m, list(ref["prompt_ids"]), max_new_tokens=20,
                           temperature=0))
    assert greedy == ref["greedy"].tolist()
    chat = m.chat_ids([{"role": "user",
                        "content": "Can I return a rain jacket?"}])
    assert chat == ref["chat_ids"].tolist()


# ---- sampling ----------------------------------------------------------

L = np.log(np.array([0.5, 0.3, 0.15, 0.05]))


def test_greedy_and_temperature_zero():
    assert sample(L, temperature=0) == 0


def test_top_k_keeps_k_tokens():
    p = probabilities(L, top_k=2)
    assert p.tolist() == pytest.approx([0.625, 0.375, 0, 0])


def test_top_p_keeps_smallest_set_reaching_p():
    # 0.5 + 0.3 = 0.8 >= 0.75, so two tokens; 0.5 alone is not enough
    assert (probabilities(L, top_p=0.75) > 0).sum() == 2
    assert (probabilities(L, top_p=0.5) > 0).sum() == 1


def test_min_p_drops_tokens_far_below_the_best():
    # 0.15 and 0.05 are under 0.4 * 0.5 = 0.2
    assert (probabilities(L, min_p=0.4) > 0).sum() == 2


def test_high_temperature_flattens():
    assert probabilities(L, temperature=100).std() < probabilities(L).std()


def test_penalties():
    out = apply_penalties(np.array([2.0, -2.0, 1.0]), [0, 1, 1],
                          repetition=2.0, presence=0.5, frequency=0.25)
    assert out.tolist() == pytest.approx([0.25, -5.0, 1.0])  # 2/2-0.75; -2*2-1


def test_sampling_follows_the_probabilities():
    rng = np.random.default_rng(0)
    draws = [sample(L, rng=rng) for _ in range(4000)]
    assert np.bincount(draws, minlength=4)[0] / 4000 == pytest.approx(0.5, abs=0.04)
