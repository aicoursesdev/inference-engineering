import json
from pathlib import Path

from overhead import decode_seconds, tiny


def test_a_tiny_decode_step_is_a_fraction_of_a_millisecond_of_pure_overhead():
    assert decode_seconds(tiny, reps=10) < 0.01
