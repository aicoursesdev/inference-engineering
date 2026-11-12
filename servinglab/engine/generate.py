"""The generation loop: forward, sample, append, repeat."""

import numpy as np

from .sampling import sample


def generate(model, prompt_ids, max_new_tokens=64, seed=None, **sampling):
    """Yield new token ids one at a time until an end token or the limit.

    `sampling` is passed to `sample`: temperature (0 = greedy), top_k, top_p,
    min_p, repetition, presence, frequency. The end token is not yielded.
    """
    rng = np.random.default_rng(seed)
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits = model.forward(ids, last_only=True)[-1]
        token = sample(logits, history=ids[len(prompt_ids):], rng=rng,
                       **sampling)
        if token in model.config.eos:
            return
        ids.append(token)
        yield token


def complete(model, text, **kwargs):
    """Text in, text out: encode, generate, decode. Handy for a first try."""
    ids = list(generate(model, model.encode(text), **kwargs))
    return model.decode(ids)
