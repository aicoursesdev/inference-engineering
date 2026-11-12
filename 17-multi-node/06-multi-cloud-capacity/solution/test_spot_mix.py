import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def output(script, *args):
    """What the script prints (it needs no network, no GPU)."""
    done = subprocess.run([sys.executable, str(HERE / script), *args],
                          capture_output=True, text=True, check=True)
    return done.stdout


def test_three_floors_each_print_a_cost_and_a_shortfall():
    rows = [l for l in output("spot_mix.py").splitlines() if l.startswith("floor")]
    assert [r.split()[1] for r in rows] == ["0", "4", "8"]
    costs = [float(r.split("$")[1].split()[0].replace(",", "")) for r in rows]
    assert costs == sorted(costs)               # more on-demand floor, more money
