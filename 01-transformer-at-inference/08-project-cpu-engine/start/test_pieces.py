"""Offline: each piece of the engine on numbers small enough to check by hand."""

import numpy as np
import pytest

import engine


def test_rmsnorm_gives_every_row_a_mean_square_of_one():
    x = np.array([[3.0, 4.0], [1.0, -1.0]])
    out = engine.rmsnorm(x, np.ones(2), 0.0)
    assert np.allclose((out * out).mean(-1), 1.0)
    assert np.allclose(out[0], np.array([3, 4]) / np.sqrt(12.5))   # (9 + 16) / 2
    assert np.allclose(engine.rmsnorm(x, np.array([2.0, 0.5]), 0.0)[1], [2.0, -0.5])


def test_silu_is_zero_at_zero_and_x_for_large_x():
    assert engine.silu(np.array(0.0)) == 0.0
    assert engine.silu(np.array(20.0)) == pytest.approx(20.0, abs=1e-6)
    assert engine.silu(np.array(1.0)) == pytest.approx(0.731059, abs=1e-5)  # 1 / (1 + e**-1)


def test_rope_leaves_position_zero_alone_and_keeps_every_length():
    x = np.random.default_rng(0).standard_normal((3, 2, 8))   # tokens, heads, head size
    assert np.allclose(engine.rope(x, np.zeros(3), 10000.0), x)
    out = engine.rope(x, np.array([0, 5, 9]), 10000.0)
    assert np.allclose(np.linalg.norm(out, axis=-1), np.linalg.norm(x, axis=-1))
    assert not np.allclose(out[1], x[1])


def test_rope_score_depends_only_on_the_gap():
    rng = np.random.default_rng(1)
    q, k = rng.standard_normal((2, 1, 1, 8))
    score = lambda pq, pk: (engine.rope(q, np.array([pq]), 100.0)
                            * engine.rope(k, np.array([pk]), 100.0)).sum()
    assert score(3, 1) == pytest.approx(score(10, 8))
    assert score(3, 1) != pytest.approx(score(10, 5))


def test_attention_first_token_sees_only_itself():
    rng = np.random.default_rng(2)
    q, k, v = rng.standard_normal((3, 4, 2, 6))
    out = engine.attention(q, k, v)
    assert out.shape == (4, 12)
    assert np.allclose(out[0], v[0].reshape(-1))


def test_attention_shares_each_key_value_head_between_query_heads():
    rng = np.random.default_rng(3)
    q = rng.standard_normal((5, 4, 6))            # 4 query heads
    k, v = rng.standard_normal((2, 5, 2, 6))      # 2 key-value heads
    out = engine.attention(q, k, v)
    same = engine.attention(q, np.repeat(k, 2, axis=1), np.repeat(v, 2, axis=1))
    assert np.allclose(out, same)


def test_attention_ignores_later_tokens():
    rng = np.random.default_rng(4)
    q, k, v = rng.standard_normal((3, 5, 2, 4))
    whole = engine.attention(q, k, v)
    assert np.allclose(whole[:3], engine.attention(q[:3], k[:3], v[:3]))
