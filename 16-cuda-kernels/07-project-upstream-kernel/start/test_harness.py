import pytest

import harness
import int4_pack


def test_summarize_reports_median_and_spread():
    s = harness.summarize([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert s["median"] == 5.5 and s["n"] == 10
    assert s["p10"] < s["median"] < s["p90"]


def test_table_has_a_speedup_column_relative_to_the_first_row():
    t = harness.markdown_table([
        {"name": "base", "median": 2.0, "p10": 1.9, "p90": 2.1},
        {"name": "ours", "median": 1.0, "p10": 0.9, "p90": 1.1}])
    assert "| ours | 1.0000 | 0.9000 | 1.1000 | 2.00x |" in t


def test_the_kernels_bit_layout_round_trips():
    import numpy as np
    q, _ = int4_pack.quantize(np.random.default_rng(0).normal(size=(2, 256)))
    assert np.array_equal(int4_pack.unpack(int4_pack.pack(q)), q)
