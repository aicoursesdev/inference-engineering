"""Your engine, in one file: the whole forward pass and the generation loop.

Loading weights and the tokenizer is given (servinglab.engine.load); everything
that computes is here. Run `make check 01.7` to compare it with a reference.
"""

import numpy as np

from servinglab.engine import load, sample  # given

__all__ = ["load", "forward", "generate"]


def rmsnorm(x, weight, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope(x, positions, theta):
    half = x.shape[-1] // 2
    freqs = theta ** (-np.arange(half) / half)
    angles = positions[:, None] * freqs[None, :]
    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    a, b = x[..., :half], x[..., half:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)


def attention(q, k, v):
    T, heads, d = q.shape
    k = np.repeat(k, heads // k.shape[1], axis=1)   # share each K/V head
    v = np.repeat(v, heads // v.shape[1], axis=1)
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(d)
    mask = np.tril(np.ones((T, T), dtype=bool))      # no peeking ahead
    scores = np.where(mask, scores, -np.inf)
    w = np.exp(scores - scores.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return np.einsum("hts,shd->thd", w, v).reshape(T, heads * d)


def block(model, i, x):
    c, w = model.config, model.weights
    p, T = f"model.layers.{i}.", len(x)
    h = rmsnorm(x, w[p + "input_layernorm.weight"], c.eps)
    q = (h @ w[p + "self_attn.q_proj.weight"].T).reshape(T, c.heads, -1)
    k = (h @ w[p + "self_attn.k_proj.weight"].T).reshape(T, c.kv_heads, -1)
    v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(T, c.kv_heads, -1)
    pos = np.arange(T)
    q, k = rope(q, pos, c.rope_theta), rope(k, pos, c.rope_theta)
    x = x + attention(q, k, v) @ w[p + "self_attn.o_proj.weight"].T
    h = rmsnorm(x, w[p + "post_attention_layernorm.weight"], c.eps)
    mlp = silu(h @ w[p + "mlp.gate_proj.weight"].T) * (h @ w[p + "mlp.up_proj.weight"].T)
    return x + mlp @ w[p + "mlp.down_proj.weight"].T


def forward(model, ids):
    """Logits, shape (len(ids), vocab): the scores for the token after each."""
    w = model.weights
    x = w["model.embed_tokens.weight"][np.asarray(ids)]
    for i in range(model.config.layers):
        x = block(model, i, x)
    x = rmsnorm(x, w["model.norm.weight"], model.config.eps)
    tied = model.config.tie_embeddings   # SmolLM2 reuses its embedding table
    return x @ w["model.embed_tokens.weight" if tied else "lm_head.weight"].T


def generate(model, prompt_ids, max_new_tokens=20, **sampling):
    """Yield new token ids: forward over everything so far, sample, repeat."""
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        token = sample(forward(model, ids)[-1], history=ids[len(prompt_ids):],
                       **sampling)
        if token in model.config.eos:
            return
        ids.append(token)
        yield token
