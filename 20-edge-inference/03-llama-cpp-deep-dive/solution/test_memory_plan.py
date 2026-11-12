from memory_plan import kv_bytes_per_token, plan
from servinglab.data import model_config


def test_qwen3_4b_cache_per_token():
    assert kv_bytes_per_token(model_config("Qwen3-4B")) == 147456


def test_ollama_default_context_is_mostly_cache():
    # `ollama ps` showed qwen3:0.6b at 4.3 GB with a 32,768-token context.
    kv, total = plan("Qwen3-0.6B", 0.40, 32768)
    assert round(kv, 2) == 3.76 and 4.0 < total < 4.4
