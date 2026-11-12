"""A fused row softmax in Triton. One program per row, the row in registers."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    raise NotImplementedError("TODO 15.3: one program per row: load with other=-inf, subtract the max, exp, divide by the sum, store, see the lesson's 'A fused softmax kernel' section")


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.stride(1) == 1
    m, n = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(m,)](out, x, x.stride(0), out.stride(0), n,
                         BLOCK_SIZE=triton.next_power_of_2(n))
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 1000)
    print("max error:", (softmax(x) - torch.softmax(x, dim=1)).abs().max().item())
