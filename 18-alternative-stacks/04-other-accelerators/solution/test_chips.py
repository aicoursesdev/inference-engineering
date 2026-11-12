import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def output(script, *args, check=True):
    done = subprocess.run([sys.executable, str(HERE / script), *args],
                          capture_output=True, text=True, check=check)
    return done


def test_h100_cannot_hold_the_cache_but_h200_can():
    rows = {l.split("  ")[0]: l for l in output("chips.py").stdout.splitlines()}
    assert rows["H100 SXM"].split()[-3] == "no"
    assert rows["H200 SXM"].split()[-3] == "yes"
    assert "1.4" in rows["H100 SXM"] and "56.4" in rows["H200 SXM"]
