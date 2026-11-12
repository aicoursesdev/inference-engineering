"""Your quality instrument: perplexity, KL divergence, top-1 agreement, a task
score and the gate (the same functions as servinglab.quality, written by you).

Every function takes arrays, so it works on logits from any engine.
`log_softmax` is given: a stable row-wise log of the softmax.
"""
import numpy as np

from servinglab.quality import log_softmax  # noqa: F401  (given)

DEFAULT_RULES = {"ppl_ratio_max": 1.05, "kl_max": 0.05,
                 "top1_min": 0.90, "task_drop_max": 0.03}


def perplexity(logits, targets):
    """exp of the average negative log-probability of the true next tokens.
    logits: (positions, vocab); targets: (positions,) token ids."""
    lp = log_softmax(logits)[np.arange(len(targets)), targets]
    return float(np.exp(-lp.mean()))


def kl_divergence(base_logits, test_logits):
    """Mean over positions of KL(base || test), in nats."""
    lp, lq = log_softmax(base_logits), log_softmax(test_logits)
    return float((np.exp(lp) * (lp - lq)).sum(axis=-1).mean())


def top1_agreement(base_logits, test_logits):
    """Share of positions where both models pick the same top token."""
    same = np.argmax(base_logits, -1) == np.argmax(test_logits, -1)
    return float(same.mean())


def accuracy(answers, golds):
    """Share of answers equal to the gold answers."""
    return sum(a == g for a, g in zip(answers, golds)) / len(golds)


def gate(base, test, rules=None):
    """Pass or fail a quantized build against the original's numbers.

    base and test are dicts with ppl and task (0 to 1 accuracy); test also
    has kl and top1. Returns (passed, reasons), one reason per rule broken."""
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
