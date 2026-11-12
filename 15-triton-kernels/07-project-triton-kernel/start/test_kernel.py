import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest

SKIP = ("Triton ships Linux wheels only: on a Mac, run this test "
        "in a free Colab CPU runtime (the lesson says how)")

import roofline


def test_ridge_points_from_the_datasheets():
    assert round(roofline.ridge(320, 65)) == 203        # T4
    assert round(roofline.ridge(3350, 989.5)) == 295    # H100 SXM


def test_attainable_is_sloped_then_flat():
    t4 = roofline.HARDWARE["T4"]
    assert roofline.attainable_tflops(1, **t4) == pytest.approx(0.32)
    assert roofline.attainable_tflops(1000, **t4) == 65.0


def test_rmsnorm_is_deep_in_the_memory_bound_region():
    m, n = 4096, 4096
    ai = roofline.rmsnorm_flops(m, n) / roofline.rmsnorm_traffic(m, n, 2)
    assert ai < 1.01 < roofline.ridge(320, 65) / 100


def test_achieved_bandwidth_of_a_known_time():
    # 2 * 4096 * 4096 * 2 bytes + weights = 67.1 MB in 0.2 ms -> ~335 GB/s
    gbs, _ = roofline.achieved(4096, 4096, 2, 0.2)
    assert gbs == pytest.approx(335.6, rel=0.01)


@pytest.mark.parametrize("m,n", [(1, 4096), (5, 576), (64, 1000)])
def test_kernel_matches_torch_in_the_interpreter(m, n):
    torch = pytest.importorskip("torch", reason=SKIP)
    pytest.importorskip("triton", reason=SKIP)
    from rmsnorm_kernel import rmsnorm, rmsnorm_torch

    torch.manual_seed(0)
    x, w = torch.randn(m, n), torch.randn(n)
    assert torch.allclose(rmsnorm(x, w, 1e-5), rmsnorm_torch(x, w, 1e-5), atol=1e-5)
