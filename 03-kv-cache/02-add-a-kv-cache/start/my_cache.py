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
    raise NotImplementedError(
        "TODO 03.2: copy the engine's block, write this step's keys and "
        "values into cache.k[i] and cache.v[i] after rope, and attend over "
        "everything stored so far, see the lesson's "
        "'What changes in the forward pass' section")


def cached_forward(model, new_ids, cache):
    """Logits for the next token, running only `new_ids` through the model."""
    raise NotImplementedError(
        "TODO 03.2: embed new_ids, give them positions that start at "
        "cache.length, run every layer through cached_block, advance "
        "cache.length and return the last row's logits, see the lesson's "
        "'What changes in the forward pass' section")


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
