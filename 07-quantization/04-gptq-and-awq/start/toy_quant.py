"""Round-to-nearest, AWQ-style scaling and GPTQ on one real layer.

    uv run --with torch --with transformers python toy_quant.py [train.txt test.txt]

With no files it calibrates on data/text/wikitext2-train-sample.txt and tests on
wikitext2-test-sample.txt (Wikipedia text, see data/README.md).

Layer 10's MLP down-projection of Qwen3-0.6B, 4-bit weights. Calibrate on
about 8,000 tokens of one text, then measure the output error on 512 tokens
of a different text, as a share of the output's own size.
"""
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from servinglab.data import DATA
from servinglab.quality import round_to_bits

LAYER, BITS = 10, 4
TEXT = DATA / "text"


def capture(model, tok, path, n):
    grab = {}
    down = model.model.layers[LAYER].mlp.down_proj
    h = down.register_forward_hook(lambda m, i, o: grab.update(x=i[0][0]))
    ids = tok(open(path).read(), return_tensors="pt", truncation=True, max_length=n)
    with torch.no_grad():
        model(**ids)
    h.remove()
    return grab["x"].numpy().astype(np.float64)


def rtn(W, g):
    return round_to_bits(W, BITS, g).astype(np.float64)


def awq(W, X, g):
    """Scale each input channel by (its mean |activation|)^alpha before rounding."""
    act, best = np.abs(X).mean(0), None
    for alpha in np.linspace(0, 1, 11):
        s = act ** alpha
        s /= np.sqrt(s.max() * s.min())              # keep scales centred
        W_hat = rtn(W * s, g) / s
        err = ((X @ W.T - X @ W_hat.T) ** 2).mean()  # on calibration data
        if best is None or err < best[0]:
            best = (err, alpha, W_hat)
    return best[2], best[1]


def gptq(W, X, g, damp=0.01):
    """Round one column at a time, pushing its error onto the columns to come."""
    W = W.copy()
    H = 2 * X.T @ X / len(X)
    H += damp * np.mean(np.diag(H)) * np.eye(len(H))
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T     # upper triangular
    top = 2 ** (BITS - 1) - 1
    Q = np.zeros_like(W)
    for j in range(W.shape[1]):
        if j % g == 0:
            scale = np.maximum(np.abs(W[:, j:j + g]).max(1), 1e-12) / top
        w = W[:, j]
        q = np.clip(np.round(w / scale), -top - 1, top) * scale
        err = (w - q) / Hinv[j, j]
        W[:, j:] -= np.outer(err, Hinv[j, j:])
        Q[:, j] = q
    return Q


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
    W = model.model.layers[LAYER].mlp.down_proj.weight.detach().numpy().astype(np.float64)
    train, test = sys.argv[1:3] or (TEXT / "wikitext2-train-sample.txt", TEXT / "wikitext2-test-sample.txt")
    Xc = capture(model, tok, train, 8192)
    Xt = capture(model, tok, test, 512)
    ref = Xt @ W.T
    rel = lambda Wq: np.sqrt(((Xt @ Wq.T - ref) ** 2).mean() / (ref ** 2).mean())
    print(f"calibration {len(Xc)} tokens, test {len(Xt)} tokens, {BITS}-bit weights")
    for g in (128, 32):
        a, alpha = awq(W, Xc, g)
        print(f"group {g:3}:  RTN {rel(rtn(W, g)):.4f}   "
              f"AWQ-style (alpha {alpha:.1f}) {rel(a):.4f}   GPTQ {rel(gptq(W, Xc, g)):.4f}")
