"""Measure acceptance rate and speedup for a draft model and for n-gram
lookup, on three kinds of text, and compare with the formula.

    uv run python acceptance.py            (about fifteen minutes on a laptop)
"""

import json
import time

from servinglab.engine import load
from servinglab.kv import generate_cached
from servinglab.speculate import (DraftModel, NgramProposer, expected_speedup,
                                  speculate)

POLICY = ("Returns: you may return rain gear within ten days of delivery if "
          "it is unworn and the tags are on. Refunds go back to the original "
          "payment method within five working days of the parcel reaching "
          "our Kochi warehouse.")
PROMPTS = {
    "open": "Explain in two sentences why the monsoon matters to farmers.",
    "copy": f"Repeat this policy word for word, changing ten days to "
            f"fourteen days.\n\n{POLICY}",
    "code": "Write a Python function that returns the n-th Fibonacci number.",
}
NEW, REPS = 32, 3
target, draft = load("SmolLM2-360M-Instruct"), load("SmolLM2-135M-Instruct")


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return time.perf_counter() - t, out


CONFIGS = (("draft model", 2), ("draft model", 4), ("n-gram", 4), ("n-gram", 8))
rows = []
for name, text in PROMPTS.items():
    ids = target.chat_ids([{"role": "user", "content": text}])
    # "plain" is speculate() with nothing guessed: the same loop and the same
    # overheads, so the speed-up compares like with like.
    jobs = {"target": lambda: list(generate_cached(target, ids, NEW,
                                                   temperature=0)),
            "draft": lambda: list(generate_cached(draft, ids, NEW,
                                                  temperature=0)),
            "plain": lambda: speculate(target, NgramProposer(), [ids], k=0,
                                       max_new_tokens=NEW)}
    for method, k in CONFIGS:
        prop = DraftModel(draft) if method == "draft model" else NgramProposer()
        jobs[method, k] = (lambda p=prop, k=k: speculate(
            target, p, [ids], k=k, max_new_tokens=NEW))
    # The laptop is busy and its speed drifts, so run every job in turn,
    # REPS times over, and keep each job's fastest run.
    best = {}
    for _ in range(REPS):
        for key, fn in jobs.items():
            secs, out = timed(fn)
            if key not in best or secs < best[key][0]:
                best[key] = (secs, out)
    want = best["target"][1]
    plain = best["plain"][1]
    base = len(plain.tokens[0]) / best["plain"][0]            # tokens/s
    step = plain.stats.verify_s / plain.stats.steps           # s per plain step
    c = (len(want) / best["target"][0]) / (len(best["draft"][1])
                                           / best["draft"][0])
    print(f"\n{name}: target {base:.2f} tok/s, draft/target step cost "
          f"c = {c:.2f}", flush=True)
    for method, k in CONFIGS:
        secs, out = best[method, k]
        s = out.stats
        v = (s.verify_s / s.steps) / step   # verify pass / a plain step
        speed = (len(out.tokens[0]) / secs) / base
        pred = expected_speedup(s.acceptance_rate, k,
                                0 if method == "n-gram" else c, v)
        rows.append(dict(prompt=name, method=method, k=k,
                         acceptance=round(s.acceptance_rate, 2),
                         tokens_per_step=round(s.tokens_per_step, 2),
                         v=round(v, 2), speedup=round(speed, 2),
                         predicted=round(pred, 2), c=round(c, 2),
                         same=out.tokens[0] == want))
        print(rows[-1], flush=True)
json.dump(rows, open("acceptance.json", "w"), indent=1)
