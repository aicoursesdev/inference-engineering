"""A KV cache for your engine: compute only the new tokens.

    from my_cache import KVCache, cached_forward, generate_cached

`cached_forward` is the engine's `forward` with one change: after the keys
and values of the new tokens are made, they go into the cache, and attention
reads ALL the keys and values from the cache instead of only the new ones.
"""

import numpy as np

from servinglab.engine.model import attention, rmsnorm, rope, silu
from servinglab.engine.sampling import sample


class KVCache:
    """Room for every layer's keys and values of one sequence."""

    def __init__(self, config, max_len):
        shape = (config.layers, max_len, config.kv_heads, config.head_dim)
        self.k = np.zeros(shape, np.float32)
        self.v = np.zeros(shape, np.float32)
        self.length = 0          # positions filled so far


def cached_block(model, i, x, positions, cache):
    """The engine's `block`, with the cache between rope and attention."""
    c, w = model.config, model.weights
    p = f"model.layers.{i}."
    h = rmsnorm(x, w[p + "input_layernorm.weight"], c.eps)
    T = len(x)
    q = (h @ w[p + "self_attn.q_proj.weight"].T).reshape(T, c.heads, c.head_dim)
    k = (h @ w[p + "self_attn.k_proj.weight"].T).reshape(T, c.kv_heads, c.head_dim)
    v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(T, c.kv_heads, c.head_dim)
    q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
    start, end = cache.length, cache.length + T
    cache.k[i, start:end], cache.v[i, start:end] = k, v        # remember them
    x = x + attention(q, cache.k[i, :end], cache.v[i, :end]) @ w[p + "self_attn.o_proj.weight"].T
    h = rmsnorm(x, w[p + "post_attention_layernorm.weight"], c.eps)
    gate = silu(h @ w[p + "mlp.gate_proj.weight"].T)
    return x + (gate * (h @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T


def cached_forward(model, new_ids, cache):
    """Logits for the next token, running only `new_ids` through the model."""
    c, w = model.config, model.weights
    new_ids = np.asarray(new_ids)
    x = w["model.embed_tokens.weight"][new_ids]
    positions = cache.length + np.arange(len(new_ids))         # where they sit in the text
    for i in range(c.layers):
        x = cached_block(model, i, x, positions, cache)
    cache.length += len(new_ids)
    x = rmsnorm(x[-1:], w["model.norm.weight"], c.eps)
    head = w["model.embed_tokens.weight" if c.tie_embeddings else "lm_head.weight"]
    return (x @ head.T)[0]


def generate_cached(model, prompt_ids, max_new_tokens=64, seed=None, **sampling):
    """Prefill the prompt once, then feed one new token per step."""
    rng = np.random.default_rng(seed)
    cache = KVCache(model.config, len(prompt_ids) + max_new_tokens)
    ids, new = list(prompt_ids), list(prompt_ids)
    for _ in range(max_new_tokens):
        logits = cached_forward(model, new, cache)
        token = sample(logits, history=ids[len(prompt_ids):], rng=rng, **sampling)
        if token in model.config.eos:
            return
        ids.append(token)
        new = [token]
        yield token
