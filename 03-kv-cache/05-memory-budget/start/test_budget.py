from servinglab.kv import memory_budget


def test_budget_adds_up():
    b = memory_budget("Qwen3-4B", "L4")
    assert abs(b["weights"] + b["activations"] + b["overhead"] + b["kv"] - b["usable"]) < 1


def test_a_16_bit_8b_model_does_not_fit_a_t4():
    assert memory_budget("Llama-3.1-8B-Instruct", "T4")["kv_tokens"] == 0


def test_more_batched_tokens_leave_less_cache():
    a = memory_budget("Qwen3-4B", "L4", max_batched_tokens=2048)["kv_tokens"]
    b = memory_budget("Qwen3-4B", "L4", max_batched_tokens=16384)["kv_tokens"]
    assert a > b
