"""Measure your own machine's two roofline numbers: bytes per second and FLOPs per second.

    uv run python machine.py          # your CPU, with NumPy
    uv run --with mlx python machine.py   # also an Apple GPU, if you have one
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def best_of(f, repeats=5):
    """The fastest of several runs: the least disturbed by other programs."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        f()
        times.append(time.perf_counter() - start)
    return min(times)


def ridge(flops_per_s, bytes_per_s):
    """FLOPs a machine can do in the time it takes to fetch one byte."""
    return flops_per_s / bytes_per_s


def cpu_bandwidth(megabytes=512, threads=None):
    """Bytes per second the CPU moves, copying one big array into another."""
    threads = threads or os.cpu_count()
    src = np.ones(megabytes * 2**20 // 4, dtype=np.float32)
    dst = np.empty_like(src)
    edges = np.linspace(0, src.size, threads + 1, dtype=int)

    def copy(k):  # NumPy lets go of Python's lock during a big copy, so threads really run together
        np.copyto(dst[edges[k]:edges[k + 1]], src[edges[k]:edges[k + 1]])

    with ThreadPoolExecutor(threads) as pool:
        seconds = best_of(lambda: list(pool.map(copy, range(threads))))
    return 2 * src.nbytes / seconds  # every byte is read once and written once


def cpu_flops(n=4096):
    """FLOPs per second of a float32 matrix multiply, which uses every core."""
    x = np.random.default_rng(0).random((n, n), dtype=np.float32)
    x @ x  # warm up
    return 2 * n**3 / best_of(lambda: x @ x)  # n*n outputs, each n multiply-adds of 2 FLOPs


def mlx_gpu():
    """An Apple GPU's bytes per second (adding two arrays) and float16 FLOPs per second."""
    import mlx.core as mx

    def timed(f):
        mx.eval(f())
        return best_of(lambda: mx.eval(f()), 10)

    a = mx.ones((2**28,), dtype=mx.float32)  # 1 GiB each
    b = mx.ones((2**28,), dtype=mx.float32)
    mx.eval(a, b)
    bandwidth = 3 * a.nbytes / timed(lambda: a + b)  # two reads and a write
    x = mx.random.normal((8192, 8192)).astype(mx.float16)
    mx.eval(x)
    return bandwidth, 2 * 8192**3 / timed(lambda: x @ x)


if __name__ == "__main__":
    rows = [("CPU (NumPy, float32)", cpu_bandwidth(), cpu_flops())]
    try:
        rows.append(("GPU (MLX, float16)", *mlx_gpu()))
    except ImportError:
        pass
    for name, bandwidth, flops in rows:
        print(f"{name}: {bandwidth / 1e9:.0f} GB/s, {flops / 1e12:.1f} TFLOPS, "
              f"ridge {ridge(flops, bandwidth):.0f} FLOPs per byte")
