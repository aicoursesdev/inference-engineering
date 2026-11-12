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
    start_m = tl.program_id(0)            # which block of query rows
    head = tl.program_id(1)
    base = head * stride_h
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptr + base + offs_m[:, None] * stride_s + offs_d[None, :],
                mask=offs_m[:, None] < seq_len, other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    hi = seq_len
    if CAUSAL:                            # skip blocks wholly in the future
        hi = tl.minimum(seq_len, (start_m + 1) * BLOCK_M)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_ptr + base + offs_n[None, :] * stride_s + offs_d[:, None],
                    mask=offs_n[None, :] < seq_len, other=0.0)   # (D, N)
        v = tl.load(v_ptr + base + offs_n[:, None] * stride_s + offs_d[None, :],
                    mask=offs_n[:, None] < seq_len, other=0.0)   # (N, D)
        s = tl.dot(q, k) * sm_scale                              # (M, N)
        s = tl.where(offs_n[None, :] < seq_len, s, float("-inf"))
        if CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)       # rescale the old sum and output
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(o_ptr + base + offs_m[:, None] * stride_s + offs_d[None, :],
             acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < seq_len)


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
