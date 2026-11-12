"""Decode attention in NumPy: one query, split the cache, merge with
log-sum-exp. This is Flash-Decoding's idea without a GPU.
"""
import numpy as np


def decode_naive(q, k, v):
    """q: (d,), k and v: (n, d) -> (d,)"""
    s = k @ q / np.sqrt(len(q))
    p = np.exp(s - s.max())
    return (p / p.sum()) @ v


def partial_attention(q, k, v):
    """Attention over one chunk of the cache: normalised output and the
    log-sum-exp of its scores, the one extra number per chunk."""
    s = k @ q / np.sqrt(len(q))
    m = s.max()
    p = np.exp(s - m)
    return (p / p.sum()) @ v, m + np.log(p.sum())


def merge(outs, lses):
    """Combine chunk outputs: each chunk counts in proportion to exp(lse)."""
    lses = np.asarray(lses)
    w = np.exp(lses - lses.max())
    return (w[:, None] * np.asarray(outs)).sum(axis=0) / w.sum()


def decode_split(q, k, v, splits):
    chunks = np.array_split(np.arange(len(k)), splits)
    parts = [partial_attention(q, k[c], v[c]) for c in chunks]
    return merge([o for o, _ in parts], [l for _, l in parts])


def programs(batch, heads, splits=1):
    """How many independent programs a decode kernel can launch."""
    return batch * heads * splits


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 1000, 64
    q, k, v = rng.normal(size=d), rng.normal(size=(n, d)), rng.normal(size=(n, d))
    ref = decode_naive(q, k, v)
    for splits in (1, 2, 8, 32):
        err = np.abs(decode_split(q, k, v, splits) - ref).max()
        print(f"{splits:>2} splits: max difference {err:.1e}")
    # Llama 3.1 8B has 32 query heads; an H100 SXM has 132 SMs.
    for splits in (1, 4, 16):
        print(f"batch 1, 32 heads, {splits:>2} splits -> "
              f"{programs(1, 32, splits)} programs for 132 SMs")
