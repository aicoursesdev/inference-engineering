import json
from pathlib import Path

import plot_curve


def test_the_recorded_sweep_draws_a_curve(tmp_path):
    data = json.loads((Path(__file__).parent / "tidewater-sweep.json").read_text())
    plot_curve.draw(data, tmp_path / "curve.png")
    assert (tmp_path / "curve.png").stat().st_size > 5000
