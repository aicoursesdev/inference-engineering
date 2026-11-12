import numpy as np

import moe_numpy as m


def test_align_pads_each_experts_group_to_the_block_size():
    ids = np.array([[1, 2, 3], [0, 1, 2], [0, 2, 3], [0, 1, 3]])
    s, e, total = m.align_block_size(ids, 4, 4)
    assert total == 16 and e.tolist() == [0, 1, 2, 3]
    assert (s == 12).sum() == 4                      # one pad slot per expert
    assert s[:4].tolist() == [3, 6, 9, 12]           # expert 0's routes, then a pad


def test_grouped_equals_naive_with_ragged_groups():
    rng = np.random.default_rng(1)
    x, W = rng.normal(size=(13, 8)), rng.normal(size=(6, 8, 5))
    ids, w = m.route(rng.normal(size=(13, 6)), 3)
    assert np.allclose(m.moe_naive(x, W, ids, w), m.moe_grouped(x, W, ids, w, block=4))


def test_warp_decode_equals_reference():
    rng = np.random.default_rng(2)
    a, Wd = rng.normal(size=(2, 40)).astype(np.float32), rng.normal(size=(5, 6, 40)).astype(np.float32)
    ids, w = np.array([3, 0]), np.array([.7, .3], np.float32)
    ref = w[0] * (Wd[3] @ a[0]) + w[1] * (Wd[0] @ a[1])
    assert np.allclose(m.warp_decode(a, Wd, ids, w), ref, atol=1e-4)


def test_one_token_reads_exactly_k_experts():
    assert m.expected_experts_touched(256, 8, 1) == 8
