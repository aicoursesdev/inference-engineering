"""A fused RMSNorm in Triton: y = x / sqrt(mean(x*x) + eps) * weight."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(out_ptr, x_ptr, w_ptr, row_stride, n_cols, eps,
                   BLOCK_SIZE: tl.constexpr):
    raise NotImplementedError("TODO 15.3: one program per row: load in float32, sqrt(sum(x*x)/n + eps), store x / rms * w, see the lesson's 'RMSNorm, the same way' section")


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    assert x.ndim == 2 and x.is_contiguous()
    m, n = x.shape
    out = torch.empty_like(x)
    rmsnorm_kernel[(m,)](out, x, weight, x.stride(0), n, eps,
                         BLOCK_SIZE=triton.next_power_of_2(n))
    return out


def rmsnorm_reference(x, weight, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
            * weight.float()).to(x.dtype)
