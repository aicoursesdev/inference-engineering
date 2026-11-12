import math

import pytest

from servinglab.cost import (api_price, breakeven, cost_per_mtok, gpus_needed,
                             hourly_rate, request_cost)


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
    assert r["input_per_mtok"] == pytest.approx(0.1)   # 0.1 s of GPU for 1,000 tokens
    assert r["output_per_mtok"] == pytest.approx(1.0)  # equals cost_per_mtok at decode speed


def test_capacity_with_headroom_and_a_spare():
    g = gpus_needed(10, 350, 120, 20000, 1000, headroom=0.3, spares=1)
    assert g["busy"] == pytest.approx(10 * (0.0175 + 0.12))
    assert g["gpus"] == math.ceil(1.375 / 0.7) + 1 == 3


def test_breakeven():
    assert breakeven(1000, 0.002) == pytest.approx(500_000)
    assert breakeven(1000, 0.002, 0.002) == math.inf


def test_lookups_carry_their_source():
    rate, row = hourly_rate("RunPod", "H100-SXM", "community cloud")
    assert rate == 2.69 and row["url"] and row["fetched"]
    (pin, pout), row = api_price("DeepInfra", "Meta-Llama-3.1-8B-Instruct-Turbo")
    assert (pin, pout) == (0.02, 0.04) and row["fetched"]
