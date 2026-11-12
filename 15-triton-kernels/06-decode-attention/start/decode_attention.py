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
    raise NotImplementedError("TODO 15.6: one program per (head, split): online softmax over its chunk, store the partial output and its lse, see the lesson's 'The two kernels' section")


@triton.jit
def decode_stage2(o_part_ptr, lse_ptr, out_ptr, n_splits,
                  HEAD_DIM: tl.constexpr, BLOCK_SPLITS: tl.constexpr):
    raise NotImplementedError("TODO 15.6: one program per head: weights exp(lse - max), merge the partial outputs, store, see the lesson's 'The two kernels' section")


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
