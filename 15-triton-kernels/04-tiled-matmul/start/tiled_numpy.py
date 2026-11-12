"""Tiled matmul in NumPy, with a count of the loads from slow memory.

Each (bm x bn) output tile is built by walking along K in steps of bk: load
a (bm x bk) tile of A and a (bk x bn) tile of B, multiply, add to a
running tile that stays in fast memory. Nothing here is fast; it counts.
"""
import numpy as np


def matmul_naive_loads(m, n, k):
    """One output number loads a row of A and a column of B: 2k values."""
    return m * n * 2 * k


def matmul_tiled(a, b, bm, bn, bk):
    raise NotImplementedError("TODO 15.4: walk bm x bn output tiles, accumulate bk-wide tiles of A and B in float32, count loads, return (c, loads), see the lesson's 'Why tiles save trips' section")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m = n = k = 256
    a = rng.normal(size=(m, k)).astype(np.float16)
    b = rng.normal(size=(k, n)).astype(np.float16)
    print("naive loads:", f"{matmul_naive_loads(m, n, k):,}")
    for bm, bn, bk in [(16, 16, 16), (32, 32, 32), (64, 64, 32), (128, 128, 32)]:
        c, loads = matmul_tiled(a, b, bm, bn, bk)
        err = np.abs(c - a.astype(np.float32) @ b.astype(np.float32)).max()
        print(f"tile {bm}x{bn}x{bk}: loads {loads:>10,}  "
              f"({matmul_naive_loads(m, n, k) / loads:.1f}x fewer)  "
              f"max error {err:.1e}")
