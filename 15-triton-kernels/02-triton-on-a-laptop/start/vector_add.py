"""Vector add in Triton. Run it on the CPU interpreter:

    TRITON_INTERPRET=1 python vector_add.py

Triton ships Linux wheels only: use a Linux laptop, WSL, or a free Colab
CPU runtime. On a Mac, open Colab.
"""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")  # before importing triton

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    raise NotImplementedError("TODO 15.2: program_id, offsets, mask, two masked loads, one masked store of x + y, see the lesson's 'Write vector add' section")


def add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024):
    out = torch.empty_like(x)
    n = out.numel()
    grid = (triton.cdiv(n, block_size),)      # one program per block
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x, y = torch.rand(98432), torch.rand(98432)
    out = add(x, y)
    print("max error:", (out - (x + y)).abs().max().item())
