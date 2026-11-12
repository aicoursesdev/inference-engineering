"""A FlashAttention-style forward pass in Triton, one head-slab per program
column. Tensors are (heads, seq, head_dim), contiguous, fp16 or fp32.

Each program owns BLOCK_M query rows, walks over the keys and values in
blocks of BLOCK_N, and keeps (m, l, acc) on chip. The (seq x seq) score
matrix is never written to memory.
"""
import math
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import triton
import triton.language as tl


@triton.jit
def attn_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, sm_scale, seq_len,
                    stride_h, stride_s,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr):
    raise NotImplementedError("TODO 15.5: load a Q block, loop over K and V blocks keeping (m_i, l_i, acc), causal mask, store acc / l_i, see the lesson's 'The Triton kernel' section")


def flash_attention(q, k, v, causal=False, block_m=32, block_n=32):
    heads, seq, dim = q.shape
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    out = torch.empty_like(q)
    grid = (triton.cdiv(seq, block_m), heads)
    attn_fwd_kernel[grid](q, k, v, out, 1.0 / math.sqrt(dim), seq,
                          q.stride(0), q.stride(1),
                          BLOCK_M=block_m, BLOCK_N=block_n,
                          HEAD_DIM=dim, CAUSAL=causal)
    return out


def attention_reference(q, k, v, causal=False):
    s = (q.float() @ k.float().transpose(1, 2)) / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[1]
        s = s.masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool)),
                          float("-inf"))
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)
