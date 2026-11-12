import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np
import pytest

SKIP = ("Triton ships Linux wheels only: on a Mac, run this test "
        "in a free Colab CPU runtime (the lesson says how)")

from tiled_numpy import matmul_naive_loads, matmul_tiled


@pytest.mark.parametrize("shape", [(64, 64, 64), (50, 70, 40)])
def test_tiled_numpy_matches_matmul_and_counts_loads(shape):
    m, n, k = shape
    rng = np.random.default_rng(1)
    a = rng.normal(size=(m, k)).astype(np.float16)
    b = rng.normal(size=(k, n)).astype(np.float16)
    c, loads = matmul_tiled(a, b, 16, 16, 16)
    assert np.allclose(c, a.astype(np.float32) @ b.astype(np.float32), atol=1e-3)
    assert loads < matmul_naive_loads(m, n, k)


def test_load_count_formula_on_exact_tiles():
    m = n = k = 64
    a = np.ones((m, k), np.float16)
    _, loads = matmul_tiled(a, a, 16, 32, 16)
    # (m/bm)(n/bn) tiles, each walks k/bk steps loading bm*bk + bk*bn
    assert loads == (64 // 16) * (64 // 32) * (64 // 16) * (16 * 16 + 16 * 32)


@pytest.mark.parametrize("shape", [(64, 64, 64), (50, 70, 40)])
def test_triton_matmul_in_the_interpreter(shape):
    torch = pytest.importorskip("torch", reason=SKIP)
    pytest.importorskip("triton", reason=SKIP)
    from tiled_matmul import matmul

    m, n, k = shape
    torch.manual_seed(0)
    a = torch.randn(m, k).half()
    b = torch.randn(k, n).half()
    ref = (a.float() @ b.float()).half()
    assert torch.allclose(matmul(a, b), ref, atol=2e-2, rtol=2e-2)
