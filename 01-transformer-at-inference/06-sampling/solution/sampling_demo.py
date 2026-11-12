"""The model's next-token scores for one prompt, and what each setting does.

    uv run python sampling_demo.py
"""

import numpy as np

from servinglab.engine import (apply_penalties, load, probabilities, sample,
                               softmax)

model = load("SmolLM2-135M-Instruct")
prompt = "After the rain stopped, we went to the"
logits = model.forward(model.encode(prompt), last_only=True)[0]


def show(title, p, n=6):
    top = [t for t in np.argsort(-p)[:n] if p[t] > 0]
    row = "  ".join(f"{model.decode([t]).strip()!r} {p[t]:.1%}" for t in top)
    print(f"{title:22s} kept {int((p > 0).sum()):5d}  {row}")


print(f"{len(logits):,} scores, one per token. Highest first:")
show("plain softmax", softmax(logits))
for t in (0.3, 1.0, 2.0):
    show(f"temperature {t}", probabilities(logits, temperature=t))
show("top-k 5", probabilities(logits, top_k=5))
show("top-p 0.9", probabilities(logits, top_p=0.9))
show("min-p 0.1", probabilities(logits, min_p=0.1))

print("\nten draws at temperature 1, seed 0 vs top-p 0.9:")
for kwargs in ({}, {"top_p": 0.9}):
    rng = np.random.default_rng(0)
    ids = [sample(logits, rng=rng, **kwargs) for _ in range(10)]
    print(kwargs or "plain", [model.decode([i]).strip() for i in ids])

print("\nrepetition penalty after ' beach' was already said:")
beach = model.encode(" beach")[0]
for pen in (1.0, 1.3, 2.0):
    p = probabilities(apply_penalties(logits, [beach], repetition=pen))
    print(f"  penalty {pen}: ' beach' has {p[beach]:.1%}")
