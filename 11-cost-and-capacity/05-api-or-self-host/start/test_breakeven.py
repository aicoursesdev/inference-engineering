import math

import pytest

from cost import breakeven


def test_breakeven():
    assert breakeven(1000, 0.002) == pytest.approx(500_000)


def test_never_when_the_api_is_no_dearer():
    assert breakeven(1000, 0.002, 0.002) == math.inf


def test_the_lessons_l4_against_deepinfra():
    per_request = (403 * 0.02 + 144 * 0.04) / 1e6
    assert round(breakeven(0.59 * 730, per_request)) == 31_164_978
