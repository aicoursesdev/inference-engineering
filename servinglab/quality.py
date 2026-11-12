"""Quality tools for quantized models: error, perplexity, KL and a gate.

Built in lesson 07.8 (the rounding helpers in 07.1 and 07.2). Pure NumPy:
every function takes arrays, so it works on logits from any engine.
"""
import numpy as np


# --- rounding weights to fewer bits (07.1, 07.2) -------------------------

def round_to_bits(w, bits, group=None):
    """Round w to signed integers of `bits` bits and back to floats.

    One scale per `group` consecutive values (the whole tensor if None):
    scale = largest absolute value / largest integer. Returns w as it would
    be after storing it in `bits` bits and reading it back.
    """
    w = np.asarray(w, dtype=np.float32)
    flat = w.reshape(-1, group) if group else w.reshape(1, -1)
    top = 2 ** (bits - 1) - 1
    scale = np.abs(flat).max(axis=1, keepdims=True) / top
    scale[scale == 0] = 1.0
    q = np.clip(np.round(flat / scale), -top - 1, top)
    return (q * scale).reshape(w.shape)


def rms_error(a, b):
    """Root-mean-square difference between two arrays."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# --- measuring a model (07.8) --------------------------------------------

def log_softmax(logits):
    """Row-wise log of the softmax, stable for large logits."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def perplexity(logits, targets):
    """exp of the average negative log-probability of the true next tokens.

    logits: (positions, vocab); targets: (positions,) token ids.
    """
    lp = log_softmax(logits)[np.arange(len(targets)), targets]
    return float(np.exp(-lp.mean()))


def kl_divergence(base_logits, test_logits):
    """Mean over positions of KL(base || test), in nats.

    How far the quantized model's next-token distribution is from the
    original's, counted where the original puts its probability.
    """
    lp, lq = log_softmax(base_logits), log_softmax(test_logits)
    return float((np.exp(lp) * (lp - lq)).sum(axis=-1).mean())


def top1_agreement(base_logits, test_logits):
    """Share of positions where both models pick the same top token."""
    same = np.argmax(base_logits, -1) == np.argmax(test_logits, -1)
    return float(same.mean())


def accuracy(answers, golds):
    """Share of answers equal to the gold answers (a small task eval)."""
    return sum(a == g for a, g in zip(answers, golds)) / len(golds)


# --- the gate (07.8) -----------------------------------------------------

DEFAULT_RULES = {"ppl_ratio_max": 1.05, "kl_max": 0.05,
                 "top1_min": 0.90, "task_drop_max": 0.03}


def gate(base, test, rules=None):
    """Pass or fail a quantized build against the original's numbers.

    base and test are dicts with ppl, task (0 to 1 accuracy); test also has
    kl and top1 (measured against the base). Returns (passed, reasons).
    """
    r = {**DEFAULT_RULES, **(rules or {})}
    why = []
    if test["ppl"] / base["ppl"] > r["ppl_ratio_max"]:
        why.append(f"perplexity {test['ppl'] / base['ppl']:.3f}x the base")
    if test["kl"] > r["kl_max"]:
        why.append(f"KL {test['kl']:.4f} over {r['kl_max']}")
    if test["top1"] < r["top1_min"]:
        why.append(f"top-1 agreement {test['top1']:.3f} under {r['top1_min']}")
    if base["task"] - test["task"] > r["task_drop_max"]:
        why.append(f"task accuracy down {base['task'] - test['task']:.3f}")
    return (not why), why
