import json

import numpy as np
import pytest

import harness
from servinglab import bench
from servinglab.data import DATA

SWEEP = DATA / "recorded" / "bench-sweep-tidewater-qwen3-0.6b.json"
DATA_ = json.loads(SWEEP.read_text())
SLO = {"ttft": 1.0, "tpot": 0.05}


def test_arrivals_start_at_zero_repeat_for_a_seed_and_have_the_asked_mean_gap():
    t = harness.arrival_times(10, 5000, seed=3)
    assert t[0] == 0 and np.array_equal(t, harness.arrival_times(10, 5000, seed=3))
    assert np.diff(t).mean() == pytest.approx(0.1, rel=0.05)


def test_burstiness_below_one_clumps_the_same_average_rate():
    poisson = np.diff(harness.arrival_times(10, 5000, burstiness=1.0))
    bursty = np.diff(harness.arrival_times(10, 5000, burstiness=0.25))
    assert bursty.mean() == pytest.approx(0.1, rel=0.1) and bursty.std() > 1.5 * poisson.std()


def test_a_request_must_meet_both_targets():
    ok = {"ok": True, "ttft": 0.5, "tpot": 0.04}
    assert harness.meets(ok, SLO)
    assert not harness.meets({**ok, "ttft": 1.5}, SLO)
    assert not harness.meets({**ok, "tpot": 0.09}, SLO)
    assert not harness.meets({"ok": False}, SLO)


def test_summarize_rebuilds_every_recorded_summary_from_its_requests():
    for run in DATA_["runs"]:
        saved = run["summary"]
        mine = harness.summarize(run["results"], saved["seconds"], saved["slo"], saved["offered_rate"])
        assert mine["goodput"] == pytest.approx(saved["goodput"])
        assert mine["slo_attainment"] == pytest.approx(saved["slo_attainment"])
        assert mine["ttft"]["p95"] == pytest.approx(saved["ttft"]["p95"])
        assert mine["output_tokens_per_s"] == pytest.approx(saved["output_tokens_per_s"])
        assert mine["max_lag"] == saved["max_lag"]


def test_capacity_is_the_highest_rate_that_met_the_slo_for_90_percent():
    summaries = [r["summary"] for r in DATA_["runs"]]
    assert harness.capacity(summaries) == 4.0
    assert harness.capacity([{**s, "slo_attainment": 0.5} for s in summaries]) is None


def test_the_report_matches_the_finished_one_and_ends_with_the_capacity_line():
    text = harness.report(DATA_)
    assert text == bench.report(DATA_)
    assert "SLO: TTFT <= 1.0 s" in text and "Capacity: 4 requests per second" in text
