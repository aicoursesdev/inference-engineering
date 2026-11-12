"""Turn the model's scores (logits) for the next token into one chosen token.

The order is the one used here (the same set of steps llama.cpp and Hugging
Face apply, though each orders them a little differently):

    penalties -> temperature -> top-k -> top-p -> min-p -> draw

`temperature=0` means greedy: take the highest score, no randomness.
"""

import numpy as np


def softmax(x):
    """Scores to probabilities. Subtracting the max keeps exp() from overflowing."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def apply_penalties(logits, history, repetition=1.0, presence=0.0,
                    frequency=0.0):
    """Push down tokens already in `history` (the ids generated so far).

    repetition: divide positive scores by it, multiply negative ones
        (the CTRL paper's rule; 1.0 does nothing).
    presence: subtract this once from every token that appeared.
    frequency: subtract this times the number of times it appeared.
    """
    logits = logits.copy()
    if len(history) == 0:
        return logits
    ids, counts = np.unique(np.asarray(history), return_counts=True)
    if repetition != 1.0:
        s = logits[ids]
        logits[ids] = np.where(s > 0, s / repetition, s * repetition)
    logits[ids] -= presence + frequency * counts
    return logits


def probabilities(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0):
    """The distribution sampling draws from, after every filter.

    Filtered-out tokens get probability 0 and the rest are renormalised.
    top_k=0 and top_p=1.0 and min_p=0.0 switch each filter off.
    """
    if temperature <= 0:
        p = np.zeros_like(logits)
        p[np.argmax(logits)] = 1.0
        return p
    logits = logits / temperature
    if top_k and top_k < len(logits):
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    p = softmax(logits)
    if min_p > 0:  # keep tokens at least min_p times as likely as the best
        p = np.where(p < min_p * p.max(), 0.0, p)
        p /= p.sum()
    if top_p < 1.0:  # keep the smallest set of top tokens with mass >= top_p
        order = np.argsort(-p, kind="stable")
        before = np.cumsum(p[order]) - p[order]  # mass ahead of each token
        keep = order[before < top_p]
        mask = np.zeros_like(p)
        mask[keep] = 1.0
        p = p * mask
        p /= p.sum()
    return p


def sample(logits, *, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
           repetition=1.0, presence=0.0, frequency=0.0, history=(),
           rng=None):
    """Pick the next token id from one position's logits (a 1-D array)."""
    logits = apply_penalties(logits, history, repetition, presence, frequency)
    p = probabilities(logits, temperature, top_k, top_p, min_p)
    if temperature <= 0:
        return int(np.argmax(p))
    rng = rng if rng is not None else np.random.default_rng()
    return int(rng.choice(len(p), p=p))
