import pytest

from cost import cost_per_mtok, request_cost


def test_one_gpu_one_speed():
    # $3.60 an hour is $0.001 a second; 1,000 tokens a second makes $1 per million tokens.
    assert cost_per_mtok(3.6, 1000) == pytest.approx(1.0)


def test_half_idle_doubles_the_price():
    assert cost_per_mtok(3.6, 1000, 0.5) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        cost_per_mtok(3.6, 1000, 0)


def test_request_cost_splits_input_and_output():
    r = request_cost(3.6, prefill_tps=10000, decode_tps=1000, input_tokens=1000, output_tokens=100)
    assert r["gpu_seconds"] == pytest.approx(0.2)
    assert r["usd"] == pytest.approx(0.0002)
    assert r["input_per_mtok"] == pytest.approx(0.1)
    assert r["output_per_mtok"] == pytest.approx(1.0)


def test_the_lessons_tidewater_request():
    r = request_cost(1.00, 907, 284, 403, 144)   # the numbers request_cost_demo.py prints
    assert round(r["gpu_seconds"], 3) == 0.951 and round(r["usd"], 6) == 0.000264
    assert round(r["input_per_mtok"], 3) == 0.306 and round(r["output_per_mtok"], 3) == 0.978
