"""Drill 2: one step of a continuous-batching scheduler with a token budget.

Rules: every running request decodes one token first; then waiting requests
are admitted in arrival order, each taking a chunk of its prompt that fits
what is left of the budget, while there is room in the batch (max_seqs)."""
from dataclasses import dataclass


@dataclass
class Req:
    name: str
    prompt: int          # prompt tokens still to prefill
    decoding: bool = False


def step(running, waiting, budget, max_seqs):
    """Return {name: tokens run this step}. Moves finished prefills to running."""
    plan = {r.name: 1 for r in running}
    left = budget - len(plan)
    for r in list(waiting):
        if left <= 0 or len(running) >= max_seqs:
            break
        n = min(r.prompt, left)
        plan[r.name] = n
        r.prompt -= n
        left -= n
        if r.prompt == 0:                 # prefill done: decodes from next step
            waiting.remove(r)
            running.append(r)
    return plan


if __name__ == "__main__":
    running = [Req("a", 0, True), Req("b", 0, True)]
    waiting = [Req("c", 900), Req("d", 50)]
    for i in range(1, 4):
        print(f"step {i}:", step(running, waiting, budget=512, max_seqs=4))
    # Checks: budget never exceeded, decodes always run, chunks add up.
    running = [Req("a", 0, True)]
    waiting = [Req("c", 900)]
    assert step(running, waiting, 512, 4) == {"a": 1, "c": 511}
    assert step(running, waiting, 512, 4) == {"a": 1, "c": 389}
    print("ok")
