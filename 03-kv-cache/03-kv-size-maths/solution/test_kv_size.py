from servinglab.data import model_config
from servinglab.kv import kv_bytes_per_token_from_json


def test_llama_8b_costs_128_kib_a_token():
    assert kv_bytes_per_token_from_json(model_config("Llama-3.1-8B-Instruct")) == 131072


def test_70b_costs_320_kib_a_token():
    assert kv_bytes_per_token_from_json(model_config("Llama-3.1-70B-Instruct")) == 327680


def test_fp8_cache_halves_it():
    cfg = model_config("Qwen3-4B")
    assert kv_bytes_per_token_from_json(cfg, 1) * 2 == kv_bytes_per_token_from_json(cfg, 2)
