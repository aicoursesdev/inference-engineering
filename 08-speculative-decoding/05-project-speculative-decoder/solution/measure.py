"""The project's measurement: plain decoding, n-gram lookup and a draft model on
three prompts, with acceptance, tokens per step and speed-up, written to the
results ledger.

    uv run python measure.py            (about ten minutes on a laptop)

The speed-up is measured against the same loop with nothing guessed, best of
three runs taken in turn, because a busy laptop's speed drifts.
"""

import time

from servinglab.engine import load
from servinglab.ledger import record
from speculate import DraftModel, NgramProposer, speculate

POLICY = ("Returns: you may return rain gear within ten days of delivery if "
          "it is unworn and the tags are on. Refunds go back to the original "
          "payment method within five working days of the parcel reaching "
          "our Kochi warehouse.")
PROMPTS = {
    "open": "Explain in two sentences why the monsoon matters to farmers.",
    "copy": "Repeat this policy word for word, changing ten days to "
            f"fourteen days.\n\n{POLICY}",
    "code": "Write a Python function that returns the n-th Fibonacci number.",
}
target, draft = load("SmolLM2-360M-Instruct"), load("SmolLM2-135M-Instruct")
CONFIGS = {"plain": (NgramProposer(), 0),
           "n-gram k=4": (NgramProposer(max_n=3), 4),
           "draft k=3": (DraftModel(draft), 3)}

print(f"{'prompt':6} {'method':11} {'accept':>7} {'tok/step':>9} {'speedup':>8}")
for name, text in PROMPTS.items():
    ids = target.chat_ids([{"role": "user", "content": text}])
    best = {}
    for _ in range(3):
        for label, (prop, k) in CONFIGS.items():
            t = time.perf_counter()
            out = speculate(target, prop, [ids], k=k, max_new_tokens=32)
            secs = time.perf_counter() - t
            if label not in best or secs < best[label][0]:
                best[label] = (secs, out)
    plain_s = best["plain"][0]
    for label, (secs, out) in best.items():
        n = len(out.tokens[0])
        s = out.stats
        print(f"{name:6} {label:11} {s.acceptance_rate:7.2f} "
              f"{s.tokens_per_step:9.2f} {plain_s / secs:7.2f}x")
        record({"lesson": "08.5", "engine": "NumPy engine + speculate.py",
                "model": "SmolLM2-360M", "hardware": "M5-Max",
                "change": f"{label}, {name} prompt", "tok_per_s": round(n / secs, 2),
                "note": f"acceptance {s.acceptance_rate:.2f}, "
                        f"{plain_s / secs:.2f}x vs plain, same loop"})
