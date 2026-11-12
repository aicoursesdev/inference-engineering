import pytest

from servinglab.data import TRACES, hardware, model_config, prices, traces

FIELDS = ["id", "arrival_s", "customer", "slo_class", "input_tokens", "output_tokens", "prefix_id", "prefix_tokens"]


@pytest.mark.parametrize("name", TRACES)
def test_traces_are_sorted_and_consistent(name):
    rows = traces(name)
    assert rows and all(list(r) == FIELDS for r in rows)
    assert [r["arrival_s"] for r in rows] == sorted(r["arrival_s"] for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(0 < r["prefix_tokens"] <= r["input_tokens"] for r in rows)


def test_trace_copies_are_independent():
    traces("nyaya")[0]["input_tokens"] = 0
    assert traces("nyaya")[0]["input_tokens"] > 0


def test_mixed_is_the_four_together():
    assert len(traces("mixed")) == sum(len(traces(n)) for n in TRACES if n != "mixed")


def test_hardware():
    h = hardware("H100-SXM")
    assert h["memory_gb"] == 80 and h["bandwidth_gbs"] == 3350
    assert h["sources"]["fp16_tflops"]["derived"] is True
    assert hardware("T4")["fp8_tflops"] is None
    assert len(hardware()) >= 12


def test_model_config():
    c = model_config("SmolLM2-135M-Instruct")
    assert (c["num_hidden_layers"], c["hidden_size"], c["num_key_value_heads"]) == (30, 576, 3)
    assert model_config("Qwen3-4B")["head_dim"] == 128


def test_prices_have_sources():
    p = prices()
    assert p["gpu_hourly"] and p["api_per_million_tokens"]
    assert all(r["url"].startswith("https://") and r["fetched"] for r in p["gpu_hourly"] + p["api_per_million_tokens"])


@pytest.mark.parametrize("call", [lambda: traces("x"), lambda: hardware("x"), lambda: model_config("x")])
def test_unknown_names_say_what_is_available(call):
    with pytest.raises(ValueError, match="choose from"):
        call()
