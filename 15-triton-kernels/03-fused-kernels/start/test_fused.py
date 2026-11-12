import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np
import pytest

SKIP = ("Triton ships Linux wheels only: on a Mac, run this test "
        "in a free Colab CPU runtime (the lesson says how)")

import traffic
from rmsnorm_numpy import rmsnorm as rmsnorm_np


def test_softmax_traffic_matches_the_triton_tutorial():
    m, n = 100, 50
    steps = traffic.softmax_unfused(m, n)
    reads = sum(r for _, r, _ in steps)
    writes = sum(w for _, _, w in steps)
    assert reads == 5 * m * n + 2 * m      # Triton tutorial 02's count
    assert writes == 3 * m * n + 2 * m
    assert 3.9 < traffic.total(steps) / traffic.total(traffic.fused(m, n)) < 4.1


def test_rmsnorm_numpy_unit_rows_stay_unit():
    x = np.random.default_rng(0).normal(size=(5, 64))
    y = rmsnorm_np(x, np.ones(64), eps=0.0)
    assert np.allclose((y * y).mean(axis=1), 1.0)


def test_rmsnorm_numpy_lesson_example():
    x = np.array([[1.0, -2.0, 3.0, -4.0]])
    y = rmsnorm_np(x, np.ones(4))
    assert np.allclose(y, x / np.sqrt(7.5 + 1e-6))   # mean of squares is 7.5


@pytest.mark.parametrize("n", [7, 576, 1000, 1024])
def test_triton_softmax_and_rmsnorm_in_the_interpreter(n):
    torch = pytest.importorskip("torch", reason=SKIP)
    pytest.importorskip("triton", reason=SKIP)
    from fused_rmsnorm import rmsnorm, rmsnorm_reference
    from fused_softmax import softmax

    torch.manual_seed(0)
    x = torch.randn(6, n) * 5
    w = torch.randn(n)
    assert torch.allclose(softmax(x), torch.softmax(x, dim=1), atol=1e-6)
    assert torch.allclose(rmsnorm(x, w), rmsnorm_reference(x, w), atol=1e-5)
