import pytest

import batch_curve


def test_the_ridge_batch_is_the_chips_ridge_scaled_by_bytes_per_weight():
    h100 = {"bandwidth_gbs": 3350, "fp16_tflops": 989.5}
    assert batch_curve.ridge_batch(2.0, h100) == pytest.approx(295.4, abs=0.1)  # 16-bit weights: batch = ridge
    assert batch_curve.ridge_batch(1.0, h100) == pytest.approx(147.7, abs=0.1)  # 8-bit weights: half


def test_decode_total_rises_with_batch_while_per_user_speed_holds_until_the_ridge():
    one, eight = (batch_curve.at(batch=b, context=160)["decode"] for b in (1, 8))
    assert eight["tokens_per_s"] > 6 * one["tokens_per_s"]
    assert eight["tokens_per_s_per_user"] > 0.8 * one["tokens_per_s_per_user"]
    assert batch_curve.at(batch=256, context=0)["decode"]["bound"] == "compute"  # no cache to read


def test_a_long_prompt_is_compute_bound_and_a_one_token_prompt_is_not():
    assert batch_curve.at(prompt=512, context=512)["prefill"]["bound"] == "compute"
    assert batch_curve.at(prompt=1, context=1)["prefill"]["bound"] == "memory"
