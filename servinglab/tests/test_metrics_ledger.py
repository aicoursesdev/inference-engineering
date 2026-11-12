import json
from pathlib import Path

import pytest

from servinglab import ledger
from servinglab.metrics import Timed, from_events, goodput, summary, throughput

# sent at 0 s; tokens at 0.2, 0.25, 0.30, 0.35: ttft 0.2, tpot 0.05
R = Timed(0.0, [0.2, 0.25, 0.30, 0.35])
SLOW = Timed(0.0, [1.5, 1.6])  # ttft 1.5 s misses a 1 s target


def test_per_request_numbers():
    assert R.ttft == pytest.approx(0.2)
    assert R.tpot == pytest.approx(0.05)  # (0.35 - 0.2) / 3
    assert R.e2e == pytest.approx(0.35)
    assert R.itls == pytest.approx([0.05] * 3)
    assert Timed(0.0, [0.3]).tpot is None


def test_throughput_and_goodput_use_one_window():
    reqs = [R, SLOW]  # window 0 -> 1.6 s, 6 tokens
    assert throughput(reqs) == pytest.approx(6 / 1.6)
    assert goodput(reqs, 1.0, 0.06) == pytest.approx(1 / 1.6)
    assert goodput(reqs, None, None) == pytest.approx(2 / 1.6)


def test_summary_good_fraction():
    s = summary([R, SLOW], (1.0, 0.05))
    assert s["good_fraction"] == 0.5 and s["requests"] == 2


def test_recorded_reply_has_33_text_chunks():
    path = Path(__file__).parents[2] / "data/recorded/llama-server-reply.json"
    req = from_events(json.loads(path.read_text())["events"])
    assert req.n_tokens == 33 and 0 < req.ttft < 0.1


def test_ledger_round_trip(tmp_path):
    f = tmp_path / "l.csv"
    ledger.record({"lesson": "00.6", "ttft_ms": 41}, f)
    ledger.record({"lesson": "01.8", "change": "engine"}, f)
    assert [r["lesson"] for r in ledger.rows(f)] == ["00.6", "01.8"]
    with pytest.raises(ValueError):
        ledger.record({"nope": 1}, f)
