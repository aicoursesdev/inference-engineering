"""Split-KV decode attention in Triton: two kernels.

Stage 1: grid (heads, splits). Each program attends one query vector to one
chunk of the cache and writes a partial output and its log-sum-exp.
Stage 2: grid (heads,). Merge the partials.

q: (heads, dim); k, v: (heads, seq, dim), contiguous. fp16 or fp32.
"""
import math
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import triton
import triton.language as tl


@triton.jit
def decode_stage1(q_ptr, k_ptr, v_ptr, o_part_ptr, lse_ptr, sm_scale,
                  seq_len, chunk, n_splits,
                  HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr):
    head = tl.program_id(0)
    split = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + head * HEAD_DIM + offs_d).to(tl.float32)

    kv_base = head * seq_len * HEAD_DIM
    lo = split * chunk
    hi = tl.minimum(lo + chunk, seq_len)
    # Scalars made from tensors, so their type is the same on every pass.
    m_i = tl.max(tl.full([BLOCK_N], float("-inf"), dtype=tl.float32), axis=0)
    l_i = tl.sum(tl.zeros([BLOCK_N], dtype=tl.float32), axis=0)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for start in range(lo, hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        valid = offs_n < hi
        k = tl.load(k_ptr + kv_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=valid[:, None], other=0.0)
        v = tl.load(v_ptr + kv_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=valid[:, None], other=0.0)
        s = tl.sum(k.to(tl.float32) * q[None, :], axis=1) * sm_scale
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    part = head * n_splits + split
    tl.store(o_part_ptr + part * HEAD_DIM + offs_d, acc / l_i)
    tl.store(lse_ptr + part, m_i + tl.log(l_i))


@triton.jit
def decode_stage2(o_part_ptr, lse_ptr, out_ptr, n_splits,
                  HEAD_DIM: tl.constexpr, BLOCK_SPLITS: tl.constexpr):
    head = tl.program_id(0)
    offs_s = tl.arange(0, BLOCK_SPLITS)
    offs_d = tl.arange(0, HEAD_DIM)
    ok = offs_s < n_splits
    lse = tl.load(lse_ptr + head * n_splits + offs_s, mask=ok,
                  other=float("-inf"))
    w = tl.exp(lse - tl.max(lse, axis=0))                 # padding gets 0
    o = tl.load(o_part_ptr + (head * n_splits + offs_s[:, None]) * HEAD_DIM
                + offs_d[None, :], mask=ok[:, None], other=0.0)
    out = tl.sum(w[:, None] * o, axis=0) / tl.sum(w, axis=0)
    tl.store(out_ptr + head * HEAD_DIM + offs_d, out.to(out_ptr.dtype.element_ty))


def decode_attention(q, k, v, splits=8, block_n=32):
    heads, seq, dim = k.shape
    splits = max(1, min(splits, triton.cdiv(seq, block_n)))
    chunk = triton.cdiv(triton.cdiv(seq, splits), block_n) * block_n
    splits = triton.cdiv(seq, chunk)          # every chunk now holds keys
    o_part = torch.empty((heads, splits, dim), dtype=torch.float32, device=q.device)
    lse = torch.empty((heads, splits), dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    decode_stage1[(heads, splits)](q, k, v, o_part, lse, 1.0 / math.sqrt(dim),
                                   seq, chunk, splits,
                                   HEAD_DIM=dim, BLOCK_N=block_n)
    decode_stage2[(heads,)](o_part, lse, out, splits, HEAD_DIM=dim,
                            BLOCK_SPLITS=triton.next_power_of_2(splits))
    return out


def decode_reference(q, k, v):
    s = (k.float() @ q.float()[:, :, None]).squeeze(-1) / math.sqrt(q.shape[-1])
    return (torch.softmax(s, dim=-1)[:, None, :] @ v.float()).squeeze(1).to(q.dtype)
