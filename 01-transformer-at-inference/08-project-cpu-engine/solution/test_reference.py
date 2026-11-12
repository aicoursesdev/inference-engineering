"""Offline: engine.py against transformers' own outputs.

The tiny model always runs. SmolLM2 itself runs once models/ has it (run
`uv run python report.py` once, which downloads it).
"""

import json
from pathlib import Path

import numpy as np
import pytest

import engine
from servinglab.engine import Config, Model
from servinglab.engine.weights import read_safetensors

ROOT = next(p for p in Path(__file__).parents if (p / "servinglab").is_dir())
DATA = ROOT / "servinglab/tests/data"
SMOL = ROOT / "models/SmolLM2-135M-Instruct/model.safetensors"


def tiny():
    d = DATA / "tiny-llama"
    config = Config.from_json(json.loads((d / "config.json").read_text()))
    return Model(config, read_safetensors(d / "model.safetensors"), None, d)


def test_tiny_logits_match_transformers():
    ref = json.loads((DATA / "tiny-llama/reference.json").read_text())
    ours = engine.forward(tiny(), ref["ids"])
    assert np.abs(ours - np.array(ref["logits"])).max() < 1e-3


@pytest.mark.skipif(not SMOL.is_file(), reason="model not downloaded yet")
def test_smollm2_logits_and_greedy_text_match_transformers():
    ref = np.load(DATA / "smollm2-135m.npz")
    model = engine.load("SmolLM2-135M-Instruct")
    last = engine.forward(model, ref["prompt_ids"])[-1]
    assert np.abs(last - ref["last_logits"]).max() < 1e-3
    out = list(engine.generate(model, list(ref["prompt_ids"]), 20,
                               temperature=0))
    assert out == ref["greedy"].tolist()
