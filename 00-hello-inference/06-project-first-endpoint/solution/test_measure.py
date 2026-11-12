"""Offline: the tool's three parts, with a fake server and a recorded reply."""

import json
import threading
import time
from pathlib import Path

import measure
from servinglab.ledger import COLUMNS, record, rows
from servinglab.metrics import Timed, from_events

RECORDING = next(p for p in Path(__file__).parents
                 if (p / "data").is_dir()) / "data/recorded/llama-server-reply.json"


def test_body_is_a_greedy_short_request_with_thinking_off():
    b = measure.body("Do you ship to Goa?", "m")
    assert b["temperature"] == 0 and b["max_tokens"] == 120 and b["model"] == "m"
    assert b["chat_template_kwargs"] == {"enable_thinking": False}
    assert [m["role"] for m in b["messages"]] == ["system", "user"]
    assert b["messages"][1]["content"] == "Do you ship to Goa?"
    assert "sentences" in b["messages"][0]["content"]
    assert len(set(measure.QUESTIONS)) == 8     # eight different questions


def test_load_warms_up_once_and_keeps_at_most_in_flight_requests_going():
    lock, state, seen = threading.Lock(), {"now": 0, "max": 0}, []

    def fake(url, body):
        with lock:
            seen.append(body)
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return Timed(0.0, [0.1, 0.2])

    bodies = [{"i": i} for i in range(10)]
    reqs = measure.load("http://x", bodies, 3, timed=fake)
    assert len(reqs) == 10                    # the warm-up's timing is dropped
    assert len(seen) == 11 and seen[0] == bodies[0]
    assert 1 < state["max"] <= 3


def test_ledger_row_scores_the_recorded_reply_and_fits_the_ledger(tmp_path):
    req = from_events(json.loads(RECORDING.read_text())["events"])
    row = measure.ledger_row([req] * 4, "Qwen3-0.6B-Q8_0", "M5-Max",
                             "baseline", 4)
    assert set(row) <= set(COLUMNS)
    assert row["lesson"] == "00.6" and row["engine"] == "llama.cpp"
    assert row["requests"] == 4 and row["tok_per_s"] > 0
    assert 20 < row["ttft_ms"] < 30 and 3 < row["tpot_ms"] < 4   # 25.8 ms, 3.13 ms
    assert "4 in flight" in row["note"] and "100%" in row["note"]
    record(row, tmp_path / "ledger.csv")
    assert rows(tmp_path / "ledger.csv")[0]["change"] == "baseline"
