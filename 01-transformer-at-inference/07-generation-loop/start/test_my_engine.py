"""Offline: your forward pass and loop against the reference outputs.

The tiny model and its logits (made with transformers) ship with the repo.
"""

import json
from pathlib import Path

import numpy as np

import my_engine
from servinglab.engine import Config, Model
from servinglab.engine.weights import read_safetensors

TINY = next(p for p in Path(__file__).parents
            if (p / "servinglab").is_dir()) / "servinglab/tests/data/tiny-llama"


def tiny():
    config = Config.from_json(json.loads((TINY / "config.json").read_text()))
    return Model(config, read_safetensors(TINY / "model.safetensors"), None, TINY)


def test_logits_match_the_reference():
    ref = json.loads((TINY / "reference.json").read_text())
    ours = my_engine.forward(tiny(), ref["ids"])
    assert np.abs(ours - np.array(ref["logits"])).max() < 1e-3


def test_greedy_loop_matches_the_shipped_engine():
    from servinglab.engine import generate
    m = tiny()
    m.config = Config(**{**m.config.__dict__, "eos": (-1,)})
    mine = list(my_engine.generate(m, [5, 17], 8, temperature=0))
    assert mine == list(generate(m, [5, 17], 8, temperature=0))
