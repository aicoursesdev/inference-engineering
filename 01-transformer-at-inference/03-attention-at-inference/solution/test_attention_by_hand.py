"""Offline: attention_by_hand.py agrees with the engine and with itself."""

import subprocess
import sys
from pathlib import Path


def test_the_hand_made_result_matches_the_engine_and_the_last_row():
    out = subprocess.run([sys.executable, "attention_by_hand.py"], cwd=Path(__file__).parent,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-300:]
    assert "matches engine: True" in out.stdout
    assert "weights after the mask and softmax" in out.stdout

