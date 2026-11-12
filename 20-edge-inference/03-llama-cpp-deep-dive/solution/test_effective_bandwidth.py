import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def test_the_ladder_turns_into_gigabytes_per_second():
    out = subprocess.run([sys.executable, str(HERE / "effective_bandwidth.py"),
                          str(HERE / "results_ladder.txt")],
                         capture_output=True, text=True, check=True).stdout
    rows = {l.split()[0]: l.split() for l in out.splitlines()[1:]}
    assert rows["Qwen3-0.6B-Q2_K"][-2:] == ["128", "28%"]
    assert rows["Qwen3-4B-Q8_0"][-2:] == ["342", "74%"]
