"""Does the accept-reject rule really give the target's distribution, even
when the draft is badly wrong? Draw 40,000 tokens and count.

    uv run python rule_check.py
"""

from collections import Counter

import numpy as np

from servinglab.speculate import accept_reject

p = np.array([0.5, 0.3, 0.15, 0.05])      # what the target wants
q = np.array([0.1, 0.1, 0.1, 0.7])        # a draft that is mostly wrong
rng = np.random.default_rng(0)
seen, kept = Counter(), 0
for _ in range(40000):
    guess = int(rng.choice(4, p=q))       # the draft guesses
    ok, token = accept_reject(p, q, guess, rng)
    seen[token] += 1
    kept += ok
print("target p      :", p)
print("tokens counted:", np.round([seen[i] / 40000 for i in range(4)], 3))
print(f"guesses kept  : {kept / 40000:.3f}   (formula: sum of min(p, q) = "
      f"{np.minimum(p, q).sum():.3f})")
