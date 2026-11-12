import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np
import pytest

SKIP = ("Triton ships Linux wheels only: on a Mac, run this test "
        "in a free Colab CPU runtime (the lesson says how)")

from split_kv import decode_naive, decode_split, merge, partial_attention, programs


@pytest.mark.parametrize("splits", [1, 2, 7, 64])
def test_split_and_merge_equals_one_pass(splits):
    rng = np.random.default_rng(3)
    q, k, v = rng.normal(size=16), rng.normal(size=(500, 16)), rng.normal(size=(500, 16))
    assert np.allclose(decode_split(q, k, v, splits), decode_naive(q, k, v))


def test_merge_weights_a_chunk_by_its_total_exp_score():
    # Two identical-score chunks, one twice as long: it must count twice.
    q = np.zeros(4)
    k = np.zeros((3, 4))
    v = np.array([[1.0] * 4, [1.0] * 4, [4.0] * 4])
    (o1, l1), (o2, l2) = partial_attention(q, k[:2], v[:2]), partial_attention(q, k[2:], v[2:])
    assert np.allclose(merge([o1, o2], [l1, l2]), v.mean(axis=0))


def test_more_splits_means_more_programs():
    assert programs(1, 32, 1) == 32 < programs(1, 32, 16) == 512


@pytest.mark.parametrize("seq", [40, 256])
def test_triton_decode_in_the_interpreter(seq):
    torch = pytest.importorskip("torch", reason=SKIP)
    pytest.importorskip("triton", reason=SKIP)
    from decode_attention import decode_attention, decode_reference

    torch.manual_seed(0)
    q = torch.randn(4, 32).half()
    k, v = torch.randn(4, seq, 32).half(), torch.randn(4, seq, 32).half()
    assert torch.allclose(decode_attention(q, k, v, splits=4),
                          decode_reference(q, k, v), atol=2e-2, rtol=2e-2)
