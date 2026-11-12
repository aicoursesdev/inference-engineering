import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np
import pytest

SKIP = ("Triton ships Linux wheels only: on a Mac, run this test "
        "in a free Colab CPU runtime (the lesson says how)")

from grid_emulator import launch


def test_emulator_covers_every_element_exactly_once():
    x = np.arange(10, dtype=np.float32)
    out, touched = launch(x, x * 10, block_size=4)
    assert np.array_equal(out, x * 11)
    seen = sorted(e for elems in touched.values() for e in elems)
    assert seen == list(range(10))
    assert touched[2] == [8, 9]          # the last block is cut by the mask


def test_triton_kernel_in_the_interpreter():
    torch = pytest.importorskip("torch", reason=SKIP)
    pytest.importorskip("triton", reason=SKIP)
    from vector_add import add

    torch.manual_seed(0)
    for n in (1, 1000, 1024, 98432):     # ragged and exact multiples
        x, y = torch.rand(n), torch.rand(n)
        assert torch.equal(add(x, y), x + y)
