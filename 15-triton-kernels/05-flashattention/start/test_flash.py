import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np
import pytest

SKIP = ("Triton ships Linux wheels only: on a Mac, run this test "
        "in a free Colab CPU runtime (the lesson says how)")

from flash_numpy import (attention_flash, attention_naive, hbm_bytes_flash,
                         hbm_bytes_naive)
from online_softmax import softmax_online, softmax_three_pass


def test_online_softmax_equals_three_pass_even_with_a_late_big_number():
    x = np.array([1.0, 2.0, 3.0, 50.0, -4.0])
    assert np.allclose(softmax_online(x)[0], softmax_three_pass(x))


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n", [16, 100])      # 100 is not a multiple of 16
def test_flash_numpy_equals_naive(n, causal):
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(n, 32)) for _ in range(3))
    assert np.allclose(attention_flash(q, k, v, 16, 16, causal),
                       attention_naive(q, k, v, causal))


def test_flash_never_pays_for_the_score_matrix():
    n, d = 16384, 64
    assert hbm_bytes_flash(n, d) < hbm_bytes_naive(n, d)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seq", [32, 70])
def test_triton_flash_attention_in_the_interpreter(seq, causal):
    torch = pytest.importorskip("torch", reason=SKIP)
    pytest.importorskip("triton", reason=SKIP)
    from flash_attention import attention_reference, flash_attention

    torch.manual_seed(0)
    q, k, v = (torch.randn(2, seq, 32).half() for _ in range(3))
    out = flash_attention(q, k, v, causal=causal)
    assert torch.allclose(out, attention_reference(q, k, v, causal),
                          atol=2e-2, rtol=2e-2)
