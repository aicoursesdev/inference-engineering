import subprocess
import sys

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_the_gap_and_the_busy_slot():
    out = run([sys.executable, "find_gap.py"])
    assert "client 1 sent at    52 ms: first token after  380.5 ms, prefill  34.0 ms -> gap  346.5 ms" in out
    assert "task  2:  1.135 to  1.531  (  396 ms)" in out
