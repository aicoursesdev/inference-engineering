import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def output(script, *args):
    """What the script prints (it needs no network, no GPU)."""
    done = subprocess.run([sys.executable, str(HERE / script), *args],
                          capture_output=True, text=True, check=True)
    return done.stdout


def test_the_three_bills():
    out = output("network.py")
    assert "23.5 ms per step" in out            # TP all-reduce, one NIC, batch 256
    assert "13.11 GB:    262 ms on one 50 GB/s NIC" in out   # Nyaya contract
    assert "440 us per layer" in out            # all-to-all over one NIC
