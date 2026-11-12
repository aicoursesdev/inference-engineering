"""Eager launches versus one CUDA graph replay, for a decode-sized step.

    python cuda_graph_decode.py        # needs an NVIDIA GPU and PyTorch (a free Colab T4 works)

Written from PyTorch's CUDA graphs documentation (the whole-network example):
warm up on a side stream, capture once with static tensors, replay with
new data copied into them. NOT RUN on the author's machine (a Mac has no CUDA).
"""

import time

import torch

assert torch.cuda.is_available(), "this file needs a CUDA GPU"
LAYERS, WIDTH = 32, 2048                     # a decode-sized stack of small matmuls
model = torch.nn.Sequential(*[
    torch.nn.Sequential(torch.nn.Linear(WIDTH, WIDTH), torch.nn.SiLU()) for _ in range(LAYERS)
]).half().cuda().eval()
static_x = torch.randn(1, WIDTH, device="cuda", dtype=torch.half)      # batch of one token


def timed(fn, reps=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


with torch.no_grad():
    eager = timed(lambda: model(static_x))

    # Warm up on a side stream, then capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            model(static_x)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_y = model(static_x)

    def replay(x=static_x):
        static_x.copy_(x)                    # new data goes into the same memory
        graph.replay()
        return static_y

    graphed = timed(replay)

print(f"eager  {eager * 1e3:6.3f} ms a step")
print(f"graph  {graphed * 1e3:6.3f} ms a step  ({eager / graphed:.1f}x faster)")
