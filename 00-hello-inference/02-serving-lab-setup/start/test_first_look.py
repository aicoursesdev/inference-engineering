"""Offline: first_look.py prints the chips, the model and the four traces."""

import subprocess
import sys
from pathlib import Path


def test_prints_four_traces_with_their_request_counts():
    out = subprocess.run([sys.executable, "first_look.py"], cwd=Path(__file__).parent,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-300:]
    for name in ("tidewater", "kodekart", "nyaya", "batch"):
        line = next(l for l in out.stdout.splitlines() if l.startswith(name))
        assert " requests, median prompt " in line and int(line.split()[1]) > 0
    assert "SmolLM2-135M: 30 layers, vector size 576" in out.stdout
