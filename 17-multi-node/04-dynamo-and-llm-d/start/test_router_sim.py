import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def output(script, *args):
    """What the script prints (it needs no network, no GPU)."""
    done = subprocess.run([sys.executable, str(HERE / script), *args],
                          capture_output=True, text=True, check=True)
    return done.stdout


def test_cache_aware_skips_about_twice_the_prefill():
    out = output("router_sim.py")
    rr = next(l for l in out.splitlines() if l.startswith("round-robin"))
    ca = next(l for l in out.splitlines() if l.startswith("cache-aware"))
    assert "42.7%" in rr and "87.8%" in ca
