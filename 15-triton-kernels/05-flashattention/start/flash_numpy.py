"""Attention three ways in NumPy, with the memory traffic each one implies.

naive: build the full (n x n) score matrix S and weights P, then P @ V.
flash: walk over K and V in blocks, keep (m, l, acc) per query row, never
       build S or P. This is what the Triton kernel does, one row-block per
       program.
"""
import numpy as np


def attention_naive(q, k, v, causal=False):
    s = q @ k.T / np.sqrt(q.shape[1])
    if causal:
        s = np.where(np.tril(np.ones_like(s, dtype=bool)), s, -np.inf)
    p = np.exp(s - s.max(axis=1, keepdims=True))
    return (p / p.sum(axis=1, keepdims=True)) @ v


def attention_flash(q, k, v, block_m=16, block_n=16, causal=False):
    raise NotImplementedError("TODO 15.5: per block of query rows keep (m, l, acc), walk K and V in blocks, rescale by exp(m_old - m_new), divide by l, see the lesson's 'Tiles of keys and values' section")


def hbm_bytes_naive(n, d, bytes_per=2):
    """Per head. Q, K, V read; S written and read; P written and read;
    O written. S and P are (n x n)."""
    return (3 * n * d + 2 * n * n + 2 * n * n + n * d) * bytes_per


def hbm_bytes_flash(n, d, bytes_per=2, block_m=128):
    """Per head. Q read once; K and V re-read once per row block; O written.
    This is the minimum a flash-style kernel pays, ignoring the causal skip."""
    row_blocks = -(-n // block_m)
    return (n * d + row_blocks * 2 * n * d + n * d) * bytes_per


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 100, 64
    q, k, v = (rng.normal(size=(n, d)) for _ in range(3))
    for causal in (False, True):
        a = attention_naive(q, k, v, causal)
        b = attention_flash(q, k, v, 16, 16, causal)
        print(f"causal={causal}: max difference {np.abs(a - b).max():.2e}")
    for d in (64, 128):
        for n in (4096, 16384):
            print(f"d={d} n={n}: S alone {n * n * 2 / 2**20:,.0f} MiB per head; "
                  f"traffic naive {hbm_bytes_naive(n, d) / 2**20:,.0f} MiB, "
                  f"flash {hbm_bytes_flash(n, d) / 2**20:,.0f} MiB")
