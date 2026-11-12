"""Offline: the cached forward pass gives the same answers as the full one."""

import json
from pathlib import Path

import numpy as np

import my_cache
from servinglab.engine import Config, Model, generate
from servinglab.engine.weights import read_safetensors

TINY = next(p for p in Path(__file__).parents
            if (p / "servinglab").is_dir()) / "servinglab/tests/data/tiny-llama"
IDS = [5, 17, 3, 99, 42, 7, 7, 61, 2, 90, 14, 33]


def tiny():
    config = Config.from_json(json.loads((TINY / "config.json").read_text()))
    return Model(config, read_safetensors(TINY / "model.safetensors"), None, TINY)


def test_prefill_matches_the_full_forward():
    m = tiny()
    cache = my_cache.KVCache(m.config, 32)
    assert np.allclose(my_cache.cached_forward(m, IDS, cache), m.forward(IDS)[-1], atol=1e-4)
    assert cache.length == len(IDS)


def test_decoding_one_token_at_a_time_matches_too():
    m = tiny()
    cache = my_cache.KVCache(m.config, 32)
    my_cache.cached_forward(m, IDS[:8], cache)
    for i in range(8, len(IDS)):
        row = my_cache.cached_forward(m, [IDS[i]], cache)
        assert np.allclose(row, m.forward(IDS[:i + 1])[-1], atol=1e-4)


def test_greedy_generation_gives_the_same_tokens():
    m = tiny()
    m.config = Config(**{**m.config.__dict__, "eos": (-1,)})
    assert (list(my_cache.generate_cached(m, IDS, 12, temperature=0))
            == list(generate(m, IDS, 12, temperature=0)))
