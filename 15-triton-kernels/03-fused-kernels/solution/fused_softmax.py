"""A fused row softmax in Triton. One program per row, the row in registers."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    # The padding must not win the max: load it as minus infinity.
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask,
                other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)             # subtract the max: no overflow
    num = tl.exp(x)                       # exp(-inf) is 0 for the padding
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)


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
