import math

import pytest

from cost import gpus_needed
from servinglab.data import traces


def test_busy_gpus_is_load_in_gpu_seconds():
    g = gpus_needed(10, 350, 120, 20000, 1000, headroom=0.3, spares=1)
    assert g["busy"] == pytest.approx(10 * (0.0175 + 0.12))
    assert g["gpus"] == math.ceil(1.375 / 0.7) + 1 == 3


def test_utilisation_is_busy_over_gpus():
    g = gpus_needed(1, 100, 100, 1000, 1000, headroom=0.0, spares=0)
    assert g["busy"] == pytest.approx(0.2) and g["gpus"] == 1 and g["utilisation"] == pytest.approx(0.2)


def test_the_lessons_kodekart_line():
    rows = traces("kodekart")
    minute = [int(r["arrival_s"] // 60) for r in rows]
    n = max(minute.count(m) for m in set(minute))
    mean_in = sum(r["input_tokens"] for r in rows) / len(rows)
    mean_out = sum(r["output_tokens"] for r in rows) / len(rows)
    g = gpus_needed(n / 60, mean_in, mean_out, 907, 284)
    assert round(g["busy"], 2) == 4.61 and g["gpus"] == 8
