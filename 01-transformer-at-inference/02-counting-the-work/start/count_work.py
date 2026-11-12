"""Matrix multiplication by hand, then the work in one SmolLM2 token.

    uv run python count_work.py
"""

import time

import numpy as np

from servinglab.data import model_config


def matmul_loops(A, B):
    """C = A @ B with three loops, counting every multiply-add."""
    m, k = len(A), len(A[0])
    n = len(B[0])
    C = [[0.0] * n for _ in range(m)]
    count = 0
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
                count += 1
    return C, count


A = [[1, 2, 3], [4, 5, 6]]          # 2 x 3
B = [[1, 0], [0, 1], [2, 2]]        # 3 x 2
C, count = matmul_loops(A, B)
print("C =", C, "after", count, "multiply-adds (2 * 3 * 2 = 12)")

n = 128
X, W = np.random.default_rng(0).random((2, n, n))
start = time.perf_counter()
matmul_loops(X.tolist(), W.tolist())
loops = time.perf_counter() - start
start = time.perf_counter()
X @ W
fast = time.perf_counter() - start
print(f"{n}x{n}: python loops {loops:.2f} s, numpy {fast * 1000:.2f} ms")

# One token through SmolLM2-135M: every weight matrix multiplies one vector.
c = model_config("SmolLM2-135M-Instruct")
d, h, kv = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
hd, ff = d // h, c["intermediate_size"]
per_layer = d * h * hd + 2 * d * kv * hd + h * hd * d + 3 * d * ff
total = c["num_hidden_layers"] * per_layer + c["vocab_size"] * d  # + output
print(f"multiply-adds per token: {total:,}  FLOPs: {2 * total:,}")
print(f"bytes of weights read:   {2 * total:,} (2 bytes each in bf16)")
