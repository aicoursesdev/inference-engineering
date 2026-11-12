import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def output(script, *args):
    """What the script prints (it needs no network, no GPU)."""
    done = subprocess.run([sys.executable, str(HERE / script), *args],
                          capture_output=True, text=True, check=True)
    return done.stdout


def test_param_count_and_gpu_counts_match_the_lesson():
    out = output("fit.py")
    assert "70.55B parameters" in out
    line = next(l for l in out.splitlines() if l.startswith("400B BF16"))
    assert line.split()[-4:] == ["16", "GPUs", "8", "GPUs"]   # H100s, then H200s
