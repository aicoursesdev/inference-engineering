"""Measure an NVIDIA GPU's real bandwidth and float16 speed, as machine.py does for a laptop.

    pip install torch        # already there on Colab
    python measure_gpu.py    # a Colab T4, or any rented card
"""

import time

import torch


def timed(f, repeats=20):
    f()  # warm up
    torch.cuda.synchronize()  # GPU work is queued: wait for it before reading the clock
    start = time.perf_counter()
    for _ in range(repeats):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


if __name__ == "__main__":
    name = torch.cuda.get_device_properties(0).name
    x = torch.empty(2**28, dtype=torch.float32, device="cuda")  # 1 GiB
    bandwidth = 2 * x.nbytes / timed(lambda: x.clone())  # a copy reads every byte and writes it
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    flops = 2 * 8192**3 / timed(lambda: a @ a)
    print(f"{name}: {bandwidth / 1e9:.0f} GB/s, {flops / 1e12:.1f} TFLOPS, "
          f"ridge {flops / bandwidth:.0f} FLOPs per byte")
