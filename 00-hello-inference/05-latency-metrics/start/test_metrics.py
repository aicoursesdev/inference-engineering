"""Offline check: the four numbers from the recorded llama-server reply."""

import json
from pathlib import Path

from servinglab.metrics import SLO, from_events, summary

RECORDING = next(p for p in Path(__file__).parents
                 if (p / "data").is_dir()) / "data/recorded/llama-server-reply.json"


def test_recorded_reply_meets_the_interactive_targets():
    req = from_events(json.loads(RECORDING.read_text())["events"])
    s = summary([req], SLO["interactive"])
    assert req.n_tokens == 33
    assert s["good_fraction"] == 1.0
    assert 0 < s["ttft_p50"] < 1.0 and 0 < s["tpot_p50"] < 0.05
