"""Drill 3: a numerically stable softmax and causal attention in NumPy."""
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)   # largest entry becomes 0: exp() cannot overflow
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_attention(q, k, v):
    """q, k, v: (tokens, head_dim). Token i may look at tokens 0..i only."""
    t, d = q.shape
    scores = q @ k.T / np.sqrt(d)
    scores = np.where(np.tril(np.ones((t, t), bool)), scores, -np.inf)
    return softmax(scores) @ v


if __name__ == "__main__":
    big = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(all="ignore"):
        naive = np.exp(big) / np.exp(big).sum()
    print("naive :", naive)
    print("stable:", softmax(big))
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((5, 4)) for _ in range(3))
    out = causal_attention(q, k, v)
    # Token 0 sees only itself, so its output is exactly v[0].
    assert np.allclose(out[0], v[0])
    # Changing a later token must not change an earlier output.
    v2 = v.copy(); v2[4] += 10
    assert np.allclose(causal_attention(q, k, v2)[:4], out[:4])
    assert np.allclose(softmax(big).sum(), 1.0)
    print("ok", out.shape)
