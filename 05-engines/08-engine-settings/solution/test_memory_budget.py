import pytest

import memory_budget


def test_qwen3_4b_on_a_t4_at_92_percent():
    b = memory_budget.budget("Qwen3-4B", "T4", 0.92, 4096)
    assert b["weights_gb"] == pytest.approx(8.04, abs=0.01)
    assert b["tokens_held"] == 45270
    assert b["full_length_requests"] == pytest.approx(11.05, abs=0.01)


def test_a_full_length_request_that_does_not_fit_shows_in_the_count():
    long_one = memory_budget.budget("Qwen3-4B", "T4", 0.92, 40960)
    assert long_one["full_length_requests"] == pytest.approx(1.11, abs=0.01)


def test_too_small_a_fraction_leaves_no_room_for_the_cache():
    assert memory_budget.budget("Qwen3-4B", "T4", 0.50, 4096)["kv_gb"] < 0
