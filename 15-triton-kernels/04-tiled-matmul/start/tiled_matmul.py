"""A tiled matmul in Triton (after Triton tutorial 03, without autotune and
without the grouped program order). fp16 in, float32 accumulate, fp16 out."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    raise NotImplementedError("TODO 15.4: build the tile pointers, loop over K with masked loads and tl.dot into a float32 acc, store as float16, see the lesson's 'The Triton kernel' section")


def matmul(a, b, block_m=32, block_n=32, block_k=32):
    assert a.shape[1] == b.shape[0] and a.dtype == b.dtype == torch.float16
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    matmul_kernel[grid](a, b, c, m, n, k,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    return c
