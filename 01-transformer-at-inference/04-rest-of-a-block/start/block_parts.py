"""Take one SmolLM2 block apart: where the weights are, and what each
piece does to a vector.

    uv run python block_parts.py
"""

import numpy as np

from servinglab.engine import load, rmsnorm, rope

model = load("SmolLM2-135M-Instruct")
w, c = model.weights, model.config

# 1. Where the parameters live
def size(prefix):
    return sum(v.size for k, v in w.items() if prefix in k)

parts = {
    "embedding table": size("embed_tokens"),
    "attention (Q, K, V, O)": size("self_attn"),
    "MLP (gate, up, down)": size("mlp"),
    "norm weights": size("layernorm") + size("model.norm"),
}
for name, n in parts.items():
    print(f"{name:24s} {n:12,d}  {n / model.n_params:5.1%}")

# 2. RMSNorm: a vector of size 4 with a big and a small number
x = np.array([[3.0, -1.0, 0.5, 2.0]])
print("\nrmsnorm:", rmsnorm(x, np.ones(4), 1e-5).round(3),
      "(mean of squares:", (x * x).mean().round(3), ")")

# 3. RoPE: the score between a query and a key depends only on how far apart
rng = np.random.default_rng(0)
q, k = rng.standard_normal((2, 1, 1, 64))
def score(pq, pk):
    a = rope(q, np.array([pq]), c.rope_theta)
    b = rope(k, np.array([pk]), c.rope_theta)
    return float((a * b).sum())
print("\nrope score, positions (3, 1):  ", round(score(3, 1), 4))
print("rope score, positions (10, 8): ", round(score(10, 8), 4), "(same gap)")
print("rope score, positions (10, 5): ", round(score(10, 5), 4), "(wider gap)")

# 4. The residual stream: each block adds a change to x, it does not replace it
from servinglab.engine.model import block

ids = model.encode("Rain gear for the monsoon")
x = w["model.embed_tokens.weight"][ids]
pos = np.arange(len(ids))
print("\n5 tokens, vector size", x.shape[1])
for i in range(3):
    y = block(model, i, x, pos)
    change = np.linalg.norm(y - x) / np.linalg.norm(x)
    print(f"block {i}: size of its change / size of x = {change:.2f}")
    x = y
