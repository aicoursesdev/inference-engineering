import json

import plot_curve
from servinglab import bench
from servinglab.data import DATA

SWEEP = DATA / "recorded" / "bench-sweep-tidewater-qwen3-0.6b.json"


def load():
    return json.loads(SWEEP.read_text())


def test_the_recorded_sweep_has_a_knee_and_a_capacity_you_can_defend():
    data = load()
    summaries = [r["summary"] for r in data["runs"]]
    assert len(summaries) >= 5 and any(s["slo_attainment"] < 0.5 for s in summaries)  # it went past failure
    assert data["capacity_rate"] == bench.capacity(summaries) == 4.0


def test_summaries_can_be_rebuilt_from_the_saved_requests():
    for run in load()["runs"]:
        s = bench.summarize(run["results"], run["summary"]["seconds"], run["summary"]["slo"], run["summary"]["offered_rate"])
        assert abs(s["goodput"] - run["summary"]["goodput"]) < 1e-9


def test_the_generator_kept_its_schedule():
    assert all(r["summary"]["max_lag"] < 0.5 for r in load()["runs"])


def test_the_report_names_its_conditions_and_the_curve_draws(tmp_path):
    text = bench.report(load())
    assert "SLO: TTFT <= 1.0 s" in text and "engine: llama.cpp" in text and "Capacity: 4 requests per second" in text
    plot_curve.draw(load(), tmp_path / "curve.png")
    assert (tmp_path / "curve.png").stat().st_size > 5000
