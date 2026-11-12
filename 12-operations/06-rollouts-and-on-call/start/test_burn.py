import subprocess
import sys

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_good_run_burns_nothing_and_bad_run_burns_two_and_a_half():
    out = run([sys.executable, "burn_rate.py"])
    assert "good:  0 of 32 over 2000 ms =  0.0%, burn rate  0.0" in out
    assert "bad :  4 of 32 over 2000 ms = 12.5%, burn rate  2.5" in out
