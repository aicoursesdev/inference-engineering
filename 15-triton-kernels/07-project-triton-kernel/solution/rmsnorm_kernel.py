"""The project kernel: a fused RMSNorm (y = x / sqrt(mean(x^2) + eps) * w)."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(out_ptr, x_ptr, w_ptr, row_stride, n_cols, eps,
             BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask,
                other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(out_ptr + row * row_stride + cols, x / rms * w, mask=mask)


def rmsnorm(x, weight, eps=1e-6):
    m, n = x.shape
    out = torch.empty_like(x)
    _rmsnorm[(m,)](out, x, weight, x.stride(0), n, eps,
                   BLOCK_SIZE=triton.next_power_of_2(n))
    return out


def rmsnorm_torch(x, weight, eps=1e-6):
    """The PyTorch version the kernel is benchmarked against."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
            * weight.float()).to(x.dtype)
