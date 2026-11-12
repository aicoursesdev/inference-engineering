import pytest

import ceiling


def test_the_recorded_run_reads_back():
    rows = ceiling.load()
    assert {r["model"] for r in rows} == {"Qwen3-0.6B", "Qwen3-1.7B"} and len(rows) == 5
    q8 = next(r for r in rows if r["file"] == "Qwen3-0.6B-Q8_0.gguf")
    assert q8["bytes"] / q8["params"] == pytest.approx(1.0625, rel=0.001)  # Q8_0 is 8.5 bits a weight


def test_a_measured_decode_never_beats_the_ceiling():
    for r in ceiling.load():
        assert r["measured"] < ceiling.ceiling(r, 406)


def test_one_fitted_overhead_predicts_the_other_models_within_ten_percent():
    rows = ceiling.load()
    anchor = next(r for r in rows if r["file"] == "Qwen3-0.6B-Q8_0.gguf")
    extra = ceiling.fit_overhead(anchor, 406)
    assert 0.001 < extra < 0.002  # about 1.4 ms a step on this laptop
    for r in rows:
        assert ceiling.ceiling(r, 406, extra) == pytest.approx(r["measured"], rel=0.10)
