import pytest

import compare
from predict_first import predictions


def laptop_rows():
    import json
    measured = json.loads(compare.RECORDED.read_text())
    sizes = [(("Qwen3-1.7B" if "1.7B" in f else "Qwen3-0.6B"), next(r for r in rs if r["n_gen"] > 0)["model_size"])
             for f, rs in measured["per_model"].items()]
    return compare.table(predictions(sizes, 406, 48.4), measured)


def test_predictions_come_from_the_roof_alone_and_are_never_beaten():
    rows = laptop_rows()
    assert len(rows) == 5
    assert all(r["decode"] < r["decode_pred"] and r["prefill"] < r["prefill_pred"] for r in rows)


def test_decode_reaches_between_a_quarter_and_all_of_its_ceiling():
    assert all(0.25 < r["decode"] / r["decode_pred"] < 1 for r in laptop_rows())


def test_the_biggest_model_gets_the_largest_share_of_decode_ceiling():
    rows = {r["file"]: r for r in laptop_rows()}
    share = lambda f: rows[f]["decode"] / rows[f]["decode_pred"]
    assert share("Qwen3-1.7B-Q8_0.gguf") > share("Qwen3-0.6B-Q8_0.gguf") > share("Qwen3-0.6B-Q2_K.gguf")


def test_prefill_lands_between_a_third_and_all_of_the_measured_peak():
    assert all(0.3 < r["prefill"] / r["prefill_pred"] < 1 for r in laptop_rows())


def test_the_fitted_overhead_is_about_a_millisecond_and_a_half():
    assert compare.fit_overhead(laptop_rows()[0]) == pytest.approx(0.00135, abs=0.0002)


def test_a_bench_row_is_flagged_noisy_when_its_repeats_disagree_by_over_ten_percent(monkeypatch):
    import record_measurements as rm
    row = {"model_type": "x", "model_size": 1, "model_n_params": 1, "n_prompt": 0, "n_gen": 128,
           "n_depth": 0, "avg_ts": 100.0}
    monkeypatch.setattr(rm, "run_json", lambda cmd: [{**row, "stddev_ts": 5.0}, {**row, "stddev_ts": 11.0}])
    quiet, noisy = rm.bench("m.gguf")
    assert not quiet["noisy"] and noisy["noisy"]
