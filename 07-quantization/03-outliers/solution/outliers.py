"""Find activation outliers in Qwen3-0.6B and see what they do to 8-bit maths.

    uv run --with torch --with transformers python outliers.py [text.txt]

With no file it reads data/text/wikitext2-test-sample.txt. Captures the
input of one MLP down-projection on 512 tokens of text, then compares
three ways of doing that matmul in 8 bits against the float result.
"""
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from servinglab.data import DATA

SAMPLE = DATA / "text" / "wikitext2-test-sample.txt"  # Wikipedia text, see data/README.md

LAYER = 10
text = open(sys.argv[1] if len(sys.argv) > 1 else SAMPLE).read()[:6000]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
grab = {}
down = model.model.layers[LAYER].mlp.down_proj
down.register_forward_hook(lambda m, i, o: grab.update(x=i[0][0].detach()))
ids = tok(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    model(**ids)
X = grab["x"].numpy().astype(np.float64)          # (tokens, 3072)
W = down.weight.detach().numpy().astype(np.float64)  # (1024, 3072)
ref = X @ W.T


def q8(a, axis):
    """Symmetric 8-bit round trip with one scale per slice along `axis`."""
    s = np.abs(a).max(axis=axis, keepdims=True) / 127
    return np.round(a / s) * s


def rel(y):
    return np.sqrt(((y - ref) ** 2).mean() / (ref ** 2).mean())


ch_max = np.abs(X).max(0)
print(f"layer {LAYER}, {X.shape[0]} tokens, {X.shape[1]} channels")
print(f"typical channel max (median) {np.median(ch_max):.3f}; "
      f"largest channel max {ch_max.max():.1f} "
      f"= {ch_max.max() / np.median(ch_max):.0f}x the median")
print("top 3 channels:", np.argsort(ch_max)[-3:][::-1].tolist())
Wq = q8(W, 1)                                     # weights: per output row
print(f"weight-only 8-bit            relative error {rel(X @ Wq.T):.4f}")
print(f"W8A8, one scale per tensor   relative error "
      f"{rel(q8(X, (0, 1)) @ Wq.T):.4f}")
print(f"W8A8, one scale per token    relative error {rel(q8(X, 1) @ Wq.T):.4f}")
for alpha in (0.5,):
    s = (ch_max ** alpha) / (np.abs(W).max(0) ** (1 - alpha))   # SmoothQuant scale
    Xs, Ws = X / s, W * s
    print(f"SmoothQuant a={alpha}, per-tensor  relative error "
          f"{rel(q8(Xs, (0, 1)) @ q8(Ws, 1).T):.4f}")
