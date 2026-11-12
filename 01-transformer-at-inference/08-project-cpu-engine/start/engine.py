"""Engine v0: your engine from lesson 01.7, the base of the serving lab.

Loading weights and the tokenizer is given (servinglab.engine.load); everything
that computes is here. `make check 01.8` compares it with transformers.

Start here: copy your finished my_engine.py from lesson 01.7 over this file.
"""

import numpy as np

from servinglab.engine import load, sample  # given

__all__ = ["load", "forward", "generate"]


def rmsnorm(x, weight, eps):
    raise NotImplementedError(
        "TODO 01.8: divide each row by the root of the mean of its squares plus eps, then multiply by weight, see the lesson's 'Build it' section")


def silu(x):
    raise NotImplementedError(
        "TODO 01.8: return x times the sigmoid of x, which is x / (1 + e to the minus x), see the lesson's 'Build it' section")


def rope(x, positions, theta):
    raise NotImplementedError(
        "TODO 01.8: rotate each pair (first half, second half) of every head by position times its frequency, see the lesson's 'Build it' section")


def attention(q, k, v):
    raise NotImplementedError(
        "TODO 01.8: share each key-value head across its query heads, score, mask, softmax and mix the values, see the lesson's 'Build it' section")


def block(model, i, x):
    raise NotImplementedError(
        "TODO 01.8: norm, queries/keys/values, RoPE, attention, add, norm, MLP, add, see the lesson's 'Build it' section")


def forward(model, ids):
    raise NotImplementedError(
        "TODO 01.8: look up the embeddings, run every block, apply the final norm and multiply by the output table (the embedding table when tied), see the lesson's 'Build it' section")


def generate(model, prompt_ids, max_new_tokens=20, **sampling):
    """Yield new token ids: forward over everything so far, sample, repeat."""
    raise NotImplementedError(
        "TODO 01.8: loop max_new_tokens times: forward over all ids so far, sample from the last row "
        "(history is the new tokens so far), stop at an end token, else append and yield it, "
        "see the lesson's 'Build it' section")
