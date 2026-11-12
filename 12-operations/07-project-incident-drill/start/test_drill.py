import subprocess, sys, json
from pathlib import Path

DIR = Path(__file__).resolve().parents[3] / "data" / "recorded" / "incidents"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def rows(name):
    return [json.loads(l) for l in (DIR / name).read_text().splitlines()]


def test_incident_1_cache_stopped_hitting():
    good, bad = rows("incident1-good.jsonl"), rows("incident1-bad.jsonl")
    assert sum(r["cache_n"] for r in good) > 20 * sum(r["cache_n"] for r in bad) and max(r["cache_n"] for r in bad) == 0
    assert sorted(r["ttft_ms"] for r in bad)[16] > 3 * sorted(r["ttft_ms"] for r in good)[16]


def test_incident_2_crashed_with_kv_not_full():
    last = rows("incident2-metrics.jsonl")[-1]
    assert last["kv_cache_usage_perc"] < 0.9 and last["gpu_free_gb"] < 0.05
    assert "OutOfMemoryError" in (DIR / "incident2-server.log").read_text()


def test_incident_3_canary_dropped():
    assert json.loads((DIR / "incident3-good-Q8_0.json").read_text())["score"] - json.loads((DIR / "incident3-bad-Q2_K.json").read_text())["score"] >= 5


def test_diagnose_runs():
    out = run([sys.executable, str(Path(__file__).with_name("diagnose.py"))])
    assert "INCIDENT 1" in out and "INCIDENT 3" in out


def test_diagnose_prints_the_percentiles():
    out = run([sys.executable, str(Path(__file__).with_name("diagnose.py")), "1"])
    assert "good TTFT p50   252.5 ms  p95   482.3 ms" in out
    assert "bad  TTFT p50  1240.6 ms  p95  2302.7 ms" in out
