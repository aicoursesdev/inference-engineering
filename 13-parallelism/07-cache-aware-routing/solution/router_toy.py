"""Round robin versus cache-aware routing, counted on Kodekart's real trace shape.

    uv run python router_toy.py

Each replica keeps the prefixes it has seen (an LRU, counted in tokens). A request's prefix
is a hit if its replica holds it. Counts tokens, no GPU; each replica keeps only a rough busy clock (40,000 prompt tokens a second).
"""
from collections import OrderedDict
from servinglab.data import traces

REPLICAS, CAPACITY, THRESHOLD = 8, 150_000, 4
trace = traces("kodekart")


class Replica:
    def __init__(self):
        self.lru, self.busy_until = OrderedDict(), 0.0
    def hit(self, r):
        return min(self.lru.get(r["prefix_id"], 0), r["prefix_tokens"])
    def serve(self, r):
        self.lru[r["prefix_id"]] = r["input_tokens"] if r["prefix_id"].startswith("kk-task") else r["prefix_tokens"]
        self.lru.move_to_end(r["prefix_id"])
        while sum(self.lru.values()) > CAPACITY:
            self.lru.popitem(last=False)
        self.busy_until = max(self.busy_until, r["arrival_s"]) + r["input_tokens"] / 40_000   # ~40k prompt tokens/s


def run(policy):
    reps, hits, total = [Replica() for _ in range(REPLICAS)], 0, 0
    for i, r in enumerate(trace):
        load = lambda x: max(0.0, x.busy_until - r["arrival_s"])
        if policy == "round robin":
            x = reps[i % REPLICAS]
        else:                                           # sticky until saturated: prefer a holder unless it is much busier
            least = min(reps, key=load)
            holders = [x for x in reps if r["prefix_id"] in x.lru]
            best = min(holders, key=load) if holders else None
            x = best if best and load(best) <= load(least) + THRESHOLD else least
        hits += x.hit(r)
        total += r["input_tokens"]
        x.serve(r)
    return hits / total


for policy in ("round robin", "cache aware"):
    print(f"{policy:12} reuses {run(policy) * 100:4.1f}% of all prompt tokens")
