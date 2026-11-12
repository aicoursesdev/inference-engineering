import subprocess
import sys

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def run_split(which):
    return run([sys.executable, "split_request.py", which])


def test_good_run_reuses_the_prompt():
    out = run_split("good")
    assert "queue and network  median   120.6 ms" in out
    assert "(284 prompt tokens computed, 2916 reused)" in out


def test_bad_run_recomputes_everything_and_prefill_grows_about_six_times():
    out = run_split("bad")
    assert "(3200 prompt tokens computed, 0 reused)" in out
    assert "queue and network  median   513.0 ms" in out
    assert "2393 waiting + 881 prefill" in out
