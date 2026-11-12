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
    raise NotImplementedError(
        "TODO 07.8: exp of the mean negative log-probability of the true next tokens, see the lesson's 'Perplexity: how surprised is the model' section")


def kl_divergence(base_logits, test_logits):
    """Mean over positions of KL(base || test), in nats."""
    raise NotImplementedError(
        "TODO 07.8: the mean over positions of KL(base || test), weighted by the base probabilities, see the lesson's 'KL divergence: how far from the original' section")


def top1_agreement(base_logits, test_logits):
    """Share of positions where both models pick the same top token."""
    raise NotImplementedError(
        "TODO 07.8: the share of positions where both models pick the same top token, see the lesson's 'KL divergence: how far from the original' section")


def accuracy(answers, golds):
    """Share of answers equal to the gold answers."""
    raise NotImplementedError(
        "TODO 07.8: the share of answers equal to the gold answers, see the lesson's 'A small task eval, and its noise' section")


def gate(base, test, rules=None):
    """Pass or fail a quantized build against the original's numbers.

    base and test are dicts with ppl and task (0 to 1 accuracy); test also
    has kl and top1. Returns (passed, reasons), one reason per rule broken."""
    raise NotImplementedError(
        "TODO 07.8: return (passed, reasons): one reason per broken rule, the four limits in DEFAULT_RULES, see the lesson's 'The gate' section")
