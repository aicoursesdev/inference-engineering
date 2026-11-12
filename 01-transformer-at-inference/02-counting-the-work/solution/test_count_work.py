"""Offline: count_work.py counts 12 multiply-adds, then the work in one token."""

import subprocess
import sys
from pathlib import Path


def test_the_small_product_costs_12_and_a_token_costs_269_million_bytes():
    out = subprocess.run([sys.executable, "count_work.py"], cwd=Path(__file__).parent,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-300:]
    assert "C = [[7.0, 8.0], [16.0, 17.0]] after 12 multiply-adds" in out.stdout
    assert "bytes of weights read:   268,959,744" in out.stdout
    assert "FLOPs: 268,959,744" in out.stdout
