"""Module 08: speculative decoding, draft and verify, on the Module 01 engine.

    from servinglab.speculate import speculate, DraftModel, NgramProposer

    draft = DraftModel(small_model)          # 08.2: a small model guesses
    # or NgramProposer(max_n=3)              # 08.2: copy from the text itself
    out = speculate(target, draft, [prompt_ids], k=4, max_new_tokens=64,
                    temperature=0)           # 08.5: batched, with timings
    out.tokens[0], out.stats.acceptance_rate, out.stats.tokens_per_step

One step, for every sequence in the batch:

    1. the proposer guesses up to k next tokens x1..xk (and, for a draft
       model, the probabilities q it drew them from);
    2. the target model runs [last token, x1..xk] ONCE, a single pass that
       gives its own probabilities p for every position;
    3. `accept_reject` walks x1..xk: keep xi with probability min(1, p/q);
       at the first miss, draw a replacement from what the target wanted
       more than the draft did, and stop; if every guess is kept, draw one
       bonus token from the target's last row;
    4. the caches are rolled back to the tokens that were kept.

Every step therefore yields 1 to k+1 tokens, and the tokens have exactly the
distribution plain sampling from the target would give (Leviathan et al.
2023, Chen et al. 2023). With temperature=0 the output is identical to
greedy decoding.

Uses the KV cache from Module 03 (`KVCache`, `step_forward`), and rolls it
back by setting `cache.length`: positions past it are overwritten later.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from .engine.sampling import probabilities
from .kv import KVCache, step_forward


# ---- 08.2: the accept-reject rule -------------------------------------------

def accept_reject(p, q, x, rng):
    """Decide one guess. p: target probabilities, q: the guesser's, x: the guess.

    Returns (kept, token). Kept with probability min(1, p[x] / q[x]); if not
    kept, the token is drawn from max(p - q, 0), renormalised (the part of the
    target's distribution the guesser under-covered). For a guesser that is
    certain (an n-gram match), q is 1 at x and 0 elsewhere.
    """
    if q[x] > 0 and rng.random() < min(1.0, p[x] / q[x]):
        return True, x
    residual = np.maximum(p - q, 0.0)
    total = residual.sum()
    residual = residual / total if total > 0 else p  # total 0 only if p == q
    return False, int(rng.choice(len(p), p=residual))


def verify(p_rows, guesses, qs, rng):
    """Walk the guesses against the target's rows.

    p_rows has len(guesses) + 1 rows of probabilities. qs[i] is the guesser's
    distribution for guess i, or None for a point mass. Returns the tokens to
    keep: the accepted guesses, then one more token (a replacement, or the
    bonus token after a clean sweep). So 1 to len(guesses) + 1 tokens.
    """
    out = []
    for i, x in enumerate(guesses):
        q = qs[i]
        if q is None:
            q = np.zeros_like(p_rows[i])
            q[x] = 1.0
        kept, token = accept_reject(p_rows[i], q, x, rng)
        out.append(token)
        if not kept:
            return out
    p = p_rows[len(guesses)]
    out.append(int(rng.choice(len(p), p=p)))
    return out


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
        for n in range(min(self.max_n, len(ids) - 1), self.min_n - 1, -1):
            tail = ids[-n:]
            for start in range(len(ids) - n - 1, -1, -1):  # latest match first
                if ids[start:start + n] == tail:
                    guess = ids[start + n:start + n + k]
                    if guess:
                        return guess, [None] * len(guess)
        return [], []

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
        m = self.model
        for i in live:
            if self.caches[i] is None:
                self.caches[i] = KVCache(m.config, len(seqs[i]) + self.room)
        caches = [self.caches[i] for i in live]
        guesses, qs = [[] for _ in live], [[] for _ in live]
        # the first call feeds everything the draft has not seen yet
        feeds = [seqs[i][c.length:] for i, c in zip(live, caches)]
        for _ in range(k):
            logits = step_forward(m, list(zip(caches, feeds)))
            for j in range(len(live)):
                q = probabilities(logits[j], temperature, **sampling)
                x = int(rng.choice(len(q), p=q))
                guesses[j].append(x)
                qs[j].append(q)
            feeds = [[g[-1]] for g in guesses]
        return guesses, qs

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
    rng = np.random.default_rng(seed)
    cfg, eos = target.config, set(target.config.eos)
    n = len(prompts)
    seqs = [list(p) for p in prompts]
    done = [False] * n
    new = [[] for _ in range(n)]
    caches = [KVCache(cfg, len(p) + max_new_tokens + k + 2) for p in prompts]
    proposer.start(n)
    stats = Stats()
    t0 = time.perf_counter()

    # prefill every prompt but its last token; that one starts the first step
    step_forward(target, [(c, s[:-1]) for c, s in zip(caches, seqs)])

    while not all(done):
        live = [i for i in range(n) if not done[i]]
        t = time.perf_counter()
        guesses, qs = proposer.propose(seqs, live, k, temperature,
                                       sampling, rng)
        stats.guess_s += time.perf_counter() - t

        t = time.perf_counter()
        rows = step_forward(target, [(caches[i], [seqs[i][-1]] + guesses[j])
                                     for j, i in enumerate(live)],
                            last_only=False)
        stats.verify_s += time.perf_counter() - t
        stats.steps += 1

        for j, i in enumerate(live):
            p = np.stack([probabilities(r, temperature, **sampling)
                          for r in rows[j]])
            kept = verify(p, guesses[j], qs[j], rng)
            stats.proposed += len(guesses[j])
            stats.accepted += len(kept) - 1  # all but the replacement or bonus
            stats.checked += min(len(kept), len(guesses[j]))
            before = len(seqs[i])
            for token in kept:
                if token in eos or len(new[i]) >= max_new_tokens:
                    done[i] = True
                    break
                seqs[i].append(token)
                new[i].append(token)
            stats.tokens += len(seqs[i]) - before
            stats.per_step.append(len(seqs[i]) - before)
            if len(new[i]) >= max_new_tokens:
                done[i] = True
            # the cache holds every committed token but the last
            caches[i].length = len(seqs[i]) - 1
            proposer.rollback(i, len(seqs[i]) - 1)
    return Result(new, stats, time.perf_counter() - t0)

