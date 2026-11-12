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
    raise NotImplementedError(
        "TODO 02.3: run f the given number of times and return the fastest time in seconds, see the lesson's 'Measure your own roof' section")


def ridge(flops_per_s, bytes_per_s):
    """FLOPs a machine can do in the time it takes to fetch one byte."""
    raise NotImplementedError(
        "TODO 02.3: return FLOPs per second divided by bytes per second, see the lesson's 'The roofline' section")


def cpu_bandwidth(megabytes=512, threads=None):
    """Bytes per second the CPU moves, copying one big array into another."""
    raise NotImplementedError(
        "TODO 02.3: copy one big array into another on every core and return bytes per second, counting each byte read and written, see the lesson's 'Measure your own roof' section")


def cpu_flops(n=4096):
    """FLOPs per second of a float32 matrix multiply, which uses every core."""
    raise NotImplementedError(
        "TODO 02.3: time an n by n float32 matrix multiply and return FLOPs per second (2 * n**3 per product), see the lesson's 'Measure your own roof' section")


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
