"""Causal attention on four tokens with two numbers each, step by step.

    uv run python attention_by_hand.py
"""

import numpy as np

from servinglab.engine import attention

np.set_printoptions(precision=2, suppress=True)

# Q, K, V: one row per token, two numbers per row (a head of size 2).
Q = np.array([[1, 0], [0, 1], [1, 1], [2, 0]], dtype=float)
K = np.array([[1, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
V = np.array([[10, 0], [0, 10], [5, 5], [20, 0]], dtype=float)

scores = Q @ K.T / np.sqrt(2)            # how well each query matches each key
print("scores (row = this token, column = token looked at):\n", scores)

mask = np.tril(np.ones((4, 4), dtype=bool))  # a token sees itself and earlier
masked = np.where(mask, scores, -np.inf)
w = np.exp(masked - masked.max(1, keepdims=True))
w /= w.sum(1, keepdims=True)             # softmax: each row sums to 1
print("weights after the mask and softmax:\n", w)
out = w @ V                              # a weighted mix of the values
print("output, one row per token:\n", out)

# The engine's own attention agrees (it expects heads: shape (T, heads, 2)).
ref = attention(Q[:, None], K[:, None], V[:, None])
print("matches engine:", np.allclose(out, ref))

# Decode: the newest token only. It needs K and V of every earlier token.
last = attention(Q[-1:, None], K[:, None], V[:, None])
print("last token alone, using all K and V:", last[0], "= row 3 above")
