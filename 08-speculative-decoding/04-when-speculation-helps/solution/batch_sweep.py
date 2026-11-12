"""How does speculation's speedup change as the batch grows?

Runs B different copy-the-policy prompts together, plain (no guessing) and
with n-gram and draft-model speculation, and prints seconds per generated
token per sequence. Writes batch_sweep.json and, if matplotlib is installed,
batch_sweep.png.

    uv run --with matplotlib python batch_sweep.py     (about twenty minutes)
"""

import json
import time

from servinglab.engine import load
from servinglab.speculate import DraftModel, NgramProposer, speculate

TEMPLATE = ("Returns: you may return rain gear within {n} days of delivery if "
            "it is unworn and the tags are on. Refunds go back to the "
            "original payment method within five working days.")
NEW, REPS, BATCHES = 24, 2, (1, 2, 4, 8, 16)
target, draft = load("SmolLM2-360M-Instruct"), load("SmolLM2-135M-Instruct")


def prompts(b):
    return [target.chat_ids([{"role": "user", "content":
            "Repeat this policy word for word, changing the days to "
            f"{n + 20}.\n\n{TEMPLATE.format(n=n + 3)}"}]) for n in range(b)]


METHODS = {"plain": lambda: (NgramProposer(), 0),
           "n-gram, k=4": lambda: (NgramProposer(max_n=3), 4),
           "draft model, k=3": lambda: (DraftModel(draft), 3)}
rows = []
for b in BATCHES:
    ps = prompts(b)
    best = {}
    for _ in range(REPS):               # interleave: the laptop's speed drifts
        for name, make in METHODS.items():
            prop, k = make()
            t = time.perf_counter()
            out = speculate(target, prop, ps, k=k, max_new_tokens=NEW)
            secs = time.perf_counter() - t
            if name not in best or secs < best[name][0]:
                best[name] = (secs, out)
    base = best["plain"][0]
    for name, (secs, out) in best.items():
        n_tok = sum(len(t) for t in out.tokens)
        rows.append(dict(batch=b, method=name, seconds=round(secs, 2),
                         ms_per_token=round(1000 * secs / (n_tok / b), 0),
                         tokens_per_s=round(n_tok / secs, 2),
                         speedup=round(base / secs, 2),
                         acceptance=round(out.stats.acceptance_rate, 2)))
        print(rows[-1], flush=True)
json.dump(rows, open("batch_sweep.json", "w"), indent=1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit
fig, ax = plt.subplots(figsize=(6, 3.6))
for name in METHODS:
    pts = [r for r in rows if r["method"] == name]
    ax.plot([r["batch"] for r in pts], [r["ms_per_token"] for r in pts],
            marker="o", label=name)
ax.set(xscale="log", yscale="log", xlabel="batch size (sequences)",
       ylabel="ms per token per sequence")
ax.set_xticks(BATCHES, [str(b) for b in BATCHES])
ax.legend()
fig.tight_layout()
fig.savefig("batch_sweep.png", dpi=150)
