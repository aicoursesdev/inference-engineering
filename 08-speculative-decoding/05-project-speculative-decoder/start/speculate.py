"""Project 08.5: your draft-and-verify decoder.

Everything but the TODOs is finished. The finished file is
servinglab/speculate.py, which Module 09 and your capstone import;
`make diff 08.5` shows what it has that this file lacks.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from servinglab.engine.sampling import probabilities
from servinglab.kv import KVCache, step_forward


# ---- 08.2: the accept-reject rule -------------------------------------------

def accept_reject(p, q, x, rng):
    """Decide one guess. p: target probabilities, q: the guesser's, x: the guess.

    Returns (kept, token). Kept with probability min(1, p[x] / q[x]); if not
    kept, the token is drawn from max(p - q, 0), renormalised (the part of the
    target's distribution the guesser under-covered). For a guesser that is
    certain (an n-gram match), q is 1 at x and 0 elsewhere.
    """
    raise NotImplementedError(
        "TODO 08.5: apply the accept-reject rule to one guess, see the lesson's 'The steps' section")


def verify(p_rows, guesses, qs, rng):
    """Walk the guesses against the target's rows.

    p_rows has len(guesses) + 1 rows of probabilities. qs[i] is the guesser's
    distribution for guess i, or None for a point mass. Returns the tokens to
    keep: the accepted guesses, then one more token (a replacement, or the
    bonus token after a clean sweep). So 1 to len(guesses) + 1 tokens.
    """
    raise NotImplementedError(
        "TODO 08.5: walk the guesses against the target's rows and return the tokens to keep, see the lesson's 'The steps' section")


def expected_tokens(alpha, k):
    """Expected tokens per step when each guess is kept with probability alpha
    (independently): 1 + alpha + ... + alpha^k = (1 - alpha^(k+1)) / (1 - alpha)."""
    return k + 1 if alpha >= 1 else (1 - alpha ** (k + 1)) / (1 - alpha)


def expected_speedup(alpha, k, c, v=1.0):
    """Leviathan et al.'s formula, with the verification pass priced. c = cost
    of one guess step / cost of one plain target step (0 for an n-gram lookup);
    v = cost of the pass that checks k + 1 tokens / cost of a plain one-token
    step (1 on a GPU at small batch, where decode waits on memory; measure it
    on a CPU, where it is more). A step costs k*c + v plain steps."""
    return expected_tokens(alpha, k) / (k * c + v)


# ---- 08.2: who guesses ------------------------------------------------------

class NgramProposer:
    """Prompt lookup: find the last `n` tokens earlier in the text and copy
    what followed. Tries n = max_n down to min_n. No model, no cost to speak
    of. Its guesses are certain, so q is None (a point mass)."""

    def __init__(self, max_n=3, min_n=1):
        self.max_n, self.min_n = max_n, min_n

    def start(self, n_seqs):
        pass

    def propose(self, seqs, live, k, temperature, sampling, rng):
        out = [self._one(seqs[i], k) for i in live]
        return [g for g, _ in out], [q for _, q in out]

    def _one(self, ids, k):
        """Guess up to k tokens for one sequence: for n from max_n down to
        min_n, find the latest earlier match of the last n tokens and return
        (the tokens that followed it, [None] * that many); ([], []) if none."""
        raise NotImplementedError(
            "TODO 08.5: find the longest recent match of the last few tokens and offer what followed, see the lesson's 'The steps' section")

    def rollback(self, i, keep):
        pass


class DraftModel:
    """A small model that guesses. It has its own KV cache per sequence and
    shares the target's tokenizer, so token ids mean the same thing."""

    def __init__(self, model, room=512):
        self.model = model
        self.room = room   # cache positions reserved beyond the prompt
        self.caches = []

    def start(self, n_seqs):
        self.caches = [None] * n_seqs

    def propose(self, seqs, live, k, temperature, sampling, rng):
        """Return (guesses, qs) for the sequences in `live`: k tokens sampled
        from this model's own probabilities, and those probabilities."""
        raise NotImplementedError(
            "TODO 08.5: give the draft its own KV cache per sequence, feed it every committed token it has not seen and sample k guesses, remembering their probabilities, see the lesson's 'The steps' section")

    def rollback(self, i, keep):
        """Forget cached positions past `keep` tokens (those were guesses
        that were not kept)."""
        c = self.caches[i]
        c.length = min(c.length, keep)


# ---- 08.5: the decoder ------------------------------------------------------

@dataclass
class Stats:
    steps: int = 0         # target passes (each one verifies a batch of guesses)
    proposed: int = 0      # guesses made
    accepted: int = 0      # guesses kept
    checked: int = 0       # guesses the target actually looked at: the kept
                           # ones, plus the first wrong one (the rest of a
                           # draft after a miss is thrown away unseen)
    tokens: int = 0        # tokens produced
    guess_s: float = 0.0   # seconds spent guessing
    verify_s: float = 0.0  # seconds spent in the target's verification passes
    per_step: list = field(default_factory=list)  # tokens each step produced

    @property
    def acceptance_rate(self):
        """Kept guesses / checked guesses: the chance a guess is kept given
        the target looked at it, the alpha of Leviathan et al.'s formula."""
        return self.accepted / self.checked if self.checked else 0.0

    @property
    def tokens_per_step(self):
        return self.tokens / self.steps if self.steps else 0.0


@dataclass
class Result:
    tokens: list   # per sequence: the new token ids
    stats: Stats
    seconds: float


def speculate(target, proposer, prompts, k=4, max_new_tokens=64, seed=0,
              temperature=0.0, **sampling):
    """Generate for a batch of prompts (lists of ids) by draft and verify.

    Stops each sequence at an end token or `max_new_tokens`. `sampling` is
    top_k, top_p or min_p, applied to target and draft alike (the guarantee
    is about the filtered distribution the target would sample from).
    """
    raise NotImplementedError(
        "TODO 08.5: the loop: prefill all but the last prompt token, then guess, one target pass with last_only=False, verify, append the kept tokens, roll the caches back, and count steps, guesses, kept and checked guesses; see the lesson's 'The steps' section")
