"""vLLM V1's scheduling idea in 25 lines: no prefill phase, no decode phase.

Each request has `computed` tokens and `total` tokens. Every step, the
scheduler hands out a token budget so computed catches up with total.
"""
BUDGET = 8


def step(reqs):
    """One scheduler step: returns {request: tokens given}."""
    left, plan = BUDGET, {}
    for r in reqs:                       # running requests first, in order
        n = min(r["total"] - r["computed"], left)
        if n > 0:
            plan[r["id"]], left = n, left - n
            r["computed"] += n
    return plan


def run():
    reqs = [{"id": "chat-A", "computed": 5, "total": 6},    # decoding
            {"id": "chat-B", "computed": 9, "total": 10},   # decoding
            {"id": "contract", "computed": 0, "total": 20}]  # a long prompt
    for t in range(1, 5):
        plan = step(reqs)
        print(f"step {t}: {plan}")
        for r in reqs:                   # a decoding request gets a new token
            if r["id"] != "contract" and r["id"] in plan:
                r["total"] += 1


if __name__ == "__main__":
    run()
