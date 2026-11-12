"""A trace-driven simulator: colocated versus disaggregated prefill and decode.

    from disagg import Cost, run_colocated, run_disagg, report

Everything is a model, so every number is a prediction, not a measurement. A step takes as
long as the slower of its memory time and its compute time (the roofline of Module 02),
each divided by an efficiency, because real kernels reach only part of the datasheet:

    mbu  share of memory bandwidth a step really uses (default 0.8)
    mfu  share of the dense FP16 peak it really uses (default 0.5)
    overhead_s  fixed time per decode step for launching work (default 1 ms)

These three are ASSUMPTIONS, chosen as typical, not measured on the reader's machine.
Simplifications, each one named so you can change it:
  * a request reserves room for its whole reply, so nothing is ever preempted;
  * the chunked colocated scheduler adds a prompt chunk's time to the decode step's time
    (a real engine shares the weight read, so this is pessimistic for it);
  * a disaggregated request's first token is emitted when its prefill ends; the KV
    transfer and the wait for a decode slot land between token 1 and token 2;
  * a replica caches shared prefixes by id (LRU, counted in tokens); the first follow-up
    step of a Kodekart task always misses, because the trace gives it a new prefix id.
"""
from collections import OrderedDict
from servinglab.data import hardware, model_config
from servinglab.roofline import shape

# Monsoon's targets (docs: inference-world): (TTFT p95 seconds, TPOT p95 seconds); batch has none.
SLO = {"interactive": (1.0, 0.05), "agent": (2.0, 0.05), "long": (8.0, 0.08), "batch": (None, None)}


def kv_bytes(layers, kv_width, tokens, bytes_per_value=2):
    """Bytes of KV cache: K and V, every layer, kv_width = kv heads x head size."""
    raise NotImplementedError("TODO 13.8: 2 (K and V) x layers x kv_width x bytes_per_value x tokens, see the lesson's 'The model' section")


class Cost:
    """How long steps take for one model on one chip (one replica, no tensor parallelism)."""

    def __init__(self, model="Llama-3.1-8B-Instruct", chip="H100-SXM", mfu=0.5, mbu=0.8, overhead_s=0.001):
        s, h = shape(model_config(model)), hardware(chip)
        self.bw, self.peak = h["bandwidth_gbs"] * 1e9 * mbu, h["fp16_tflops"] * 1e12 * mfu
        self.mem_bytes = h["memory_gb"] * 1e9
        self.weights = s["params"] * 2
        self.mm_bytes = s["matmul_params"] * 2
        self.mm_params, self.kv_tok = s["matmul_params"], s["kv_bytes_per_token_f16"]
        self.attn_dim = s["layers"] * s["heads"] * s["head_dim"]
        self.overhead_s = overhead_s
        self.kv_capacity = int((0.9 * self.mem_bytes - self.weights) / self.kv_tok)  # tokens a replica can hold

    def prefill_s(self, n, cached=0):
        """Time to prefill a prompt of n tokens when the first `cached` already have their KV."""
        raise NotImplementedError("TODO 13.8: the slower of memory time and compute time for the uncached tokens: 35,000 tokens should take about 1.7 s, see the lesson's 'Steps' section")

    def decode_s(self, batch, total_ctx):
        """Time for one decode step of `batch` requests holding `total_ctx` tokens of KV between them."""
        raise NotImplementedError("TODO 13.8: the slower of memory time (weights plus KV, over bandwidth) and compute time (matmul and attention FLOPs over peak), plus overhead_s, see the lesson's 'The model' section")


class PrefixCache:
    """An LRU of shared prefixes, counted in tokens."""

    def __init__(self, capacity):
        self.capacity, self.items = capacity, OrderedDict()

    def lookup(self, r):
        n = self.items.get(r["prefix_id"], 0)
        if n:
            self.items.move_to_end(r["prefix_id"])
        return min(n, r["prefix_tokens"])

    def store(self, r):
        self.items[r["prefix_id"]] = r["input_tokens"] if r["prefix_id"].startswith("kk-task") else r["prefix_tokens"]
        self.items.move_to_end(r["prefix_id"])
        while sum(self.items.values()) > self.capacity:
            self.items.popitem(last=False)


class Replica:
    """One GPU: continuous batching, plus (when `prefill` is true) the prompts too."""

    def __init__(self, cost, prefill=True, policy="prefill_first", chunk=512, max_batch=256, cache_tokens=0):
        self.c, self.prefill, self.policy, self.chunk, self.max_batch = cost, prefill, policy, chunk, max_batch
        self.t, self.waiting, self.ready, self.running, self.held = 0.0, [], [], [], 0
        self.cache = PrefixCache(cache_tokens)

    def load(self):
        return len(self.waiting) + len(self.ready) + len(self.running)

    def busy(self):
        return bool(self.waiting or self.ready or self.running)

    def _fits(self, r):
        return self.held + r["input_tokens"] + r["output_tokens"] <= self.c.kv_capacity and len(self.running) < self.max_batch

    def _admit_ready(self):
        while self.ready and self.ready[0]["decode_arrival"] <= self.t and self._fits(self.ready[0]):
            r = self.ready.pop(0)
            self.held += r["input_tokens"] + r["output_tokens"]
            r["left"], self.running = r["output_tokens"] - 1, self.running + [r]

    def _finish(self):
        for r in [r for r in self.running if r["left"] <= 0]:
            r["done_t"] = self.t
            self.held -= r["input_tokens"] + r["output_tokens"]
            self.running.remove(r)

    def _decode(self):
        ctx = sum(r["input_tokens"] + r["output_tokens"] - r["left"] for r in self.running)
        self.t += self.c.decode_s(len(self.running), ctx)
        for r in self.running:
            r["left"] -= 1
        self._finish()

    def step(self):
        """Advance by one iteration. Returns False if there was nothing to do yet."""
        self._admit_ready()
        head = self.waiting[0] if self.waiting and self._fits(self.waiting[0]) else None
        if head and self.prefill:
            if "cached" not in head:
                head["cached"] = self.cache.lookup(head)
                head["todo"] = head["input_tokens"] - head["cached"]
            if self.policy == "prefill_first":
                self.t += self.c.prefill_s(head["input_tokens"], head["cached"])
                head["todo"] = 0
            else:  # chunked: decode step plus one chunk of the prompt, in one iteration
                n = min(self.chunk, head["todo"])
                done = head["input_tokens"] - head["todo"]
                dt = self.c.prefill_s(done + n, done) if n else 0.0
                if self.running:
                    ctx = sum(r["input_tokens"] + r["output_tokens"] - r["left"] for r in self.running)
                    dt += self.c.decode_s(len(self.running), ctx)
                    for r in self.running:
                        r["left"] -= 1
                self.t += dt
                head["todo"] -= n
                self._finish()
            if head["todo"] <= 0:
                self.waiting.pop(0)
                self.cache.store(head)
                head["first_t"] = self.t
                self.held += head["input_tokens"] + head["output_tokens"]
                head["left"] = head["output_tokens"] - 1
                self.running.append(head)
                self._finish()
            return True
        if self.running:
            self._decode()
            return True
        return False


def _route(policy, replicas, r, i, threshold=8):
    """Pick a replica: round robin, least loaded, or cache aware (sticky until it is busy)."""
    if policy == "round_robin":
        return replicas[i % len(replicas)]
    least = min(replicas, key=lambda x: x.load())
    if policy == "least_loaded":
        return least
    holders = [x for x in replicas if r["prefix_id"] in x.cache.items]
    if holders:
        best = min(holders, key=lambda x: x.load())
        if best.load() <= least.load() + threshold:
            return best
    return least


def _drive(replicas, arrivals, deliver):
    """Run replicas against time-ordered arrivals; deliver(r, i) hands one to a replica."""
    i = 0
    while i < len(arrivals) or any(x.busy() for x in replicas):
        active = [x for x in replicas if x.busy()]
        horizon = min(x.t for x in active) if active else (arrivals[i]["_at"] if i < len(arrivals) else 0)
        while i < len(arrivals) and arrivals[i]["_at"] <= max(horizon, 0):
            deliver(arrivals[i], i)
            i += 1
        active = [x for x in replicas if x.busy()]
        if not active:
            continue
        x = min(active, key=lambda x: x.t)
        if not x.step():  # only waiting for a decode slot or a transfer to land: jump to the next event
            nxt = min([r["decode_arrival"] for r in x.ready] + [arrivals[i]["_at"] if i < len(arrivals) else 1e18])
            x.t = max(x.t, nxt) if nxt < 1e18 else x.t + 1e-3


def _fresh(trace, scale):
    t0 = trace[0]["arrival_s"]
    out = []
    for r in trace:
        r = dict(r)
        r["_at"] = (r["arrival_s"] - t0) / scale  # `scale` times faster than the real day
        out.append(r)
    return out


def run_colocated(trace, gpus, cost, scale=1.0, policy="prefill_first", router="least_loaded", cache_tokens=0):
    reqs = _fresh(trace, scale)
    reps = [Replica(cost, True, policy, cache_tokens=cache_tokens) for _ in range(gpus)]

    def deliver(r, i):
        x = _route(router, reps, r, i)
        x.t = max(x.t, r["_at"]) if not x.busy() else x.t
        x.waiting.append(r)

    _drive(reps, reqs, deliver)
    return reqs


def run_disagg(trace, prefill_gpus, decode_gpus, cost, link_bytes_per_s, scale=1.0, router="least_loaded",
               cache_tokens=0, link_latency_s=0.0005):
    reqs = _fresh(trace, scale)
    pre = [{"free": 0.0, "cache": PrefixCache(cache_tokens)} for _ in range(prefill_gpus)]
    for i, r in enumerate(reqs):  # prefill replicas are single-request FIFO servers: solve them directly
        class _V:  # a view so _route can read load and cache
            def __init__(s, p): s.p, s.cache = p, p["cache"]
            def load(s): return max(0.0, s.p["free"] - r["_at"]) * 100
        views = [_V(p) for p in pre]
        p = _route(router, views, r, i, threshold=10).p
        cached = p["cache"].lookup(r)
        start = max(p["free"], r["_at"])
        p["free"] = start + cost.prefill_s(r["input_tokens"], cached)
        p["cache"].store(r)
        r["first_t"] = p["free"]
        raise NotImplementedError("TODO 13.8: transfer_s is latency + KV bytes / link speed, see the lesson's 'Steps' section")
        r["decode_arrival"] = r["first_t"] + r["transfer_s"]
    dec = [Replica(cost, prefill=False) for _ in range(decode_gpus)]
    order = sorted(reqs, key=lambda r: r["decode_arrival"])
    for r in order:
        r["_at"] = r["decode_arrival"]

    def deliver(r, i):
        x = min(dec, key=lambda d: d.load())
        x.t = max(x.t, r["_at"]) if not x.busy() else x.t
        x.ready.append(r)

    _drive(dec, order, deliver)
    for r in reqs:
        r["_at"] = (r["arrival_s"] - trace[0]["arrival_s"]) / scale
    return reqs


def report(reqs, slo=SLO):
    """Goodput (requests per second that met both targets), attainment and p95s."""
    raise NotImplementedError("TODO 13.8: goodput (requests meeting both targets, per second of run), attained share and the p95 first-token and per-token times, see the lesson's 'Steps' section")
