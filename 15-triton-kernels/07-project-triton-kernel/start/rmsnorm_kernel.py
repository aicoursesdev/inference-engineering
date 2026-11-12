"""The project kernel: a fused RMSNorm (y = x / sqrt(mean(x^2) + eps) * w)."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(out_ptr, x_ptr, w_ptr, row_stride, n_cols, eps,
             BLOCK_SIZE: tl.constexpr):
    raise NotImplementedError("TODO 15.7: one program per row: load in float32, sqrt(sum(x*x)/n + eps), store x / rms * w, see the lesson's 'Steps' section")


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
