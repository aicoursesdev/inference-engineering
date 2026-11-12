"""Module 04: the continuous-batching scheduler, and a simulator to run it.

    from servinglab.scheduler import Scheduler, Request, simulate, StepModel

    sched = Scheduler(num_blocks=4000, token_budget=512, chunked=True)
    results = simulate(rows, sched, StepModel("Qwen3-4B", "L4"))      # 04.6

One step of an engine is: ask the scheduler what to run, run it, tell the
scheduler what happened.

    plan = sched.schedule(now)         # [(request, n_tokens), ...]
    ...run those tokens (a real engine) or price them (the simulator)...
    sched.update(plan, now_after)

The scheduler never touches tensors. It knows each request's tokens, how many
of them have a computed key and value (`computed`), and which blocks the
allocator (Module 03) gave it. A request with `computed` behind its tokens is
still prefilling; one that is level has just one new token to decode.

    mode "static"       a batch forms, runs to its slowest member, then the next
                        forms (04.1);
    mode "continuous"   requests join and leave at every step (04.2). With
                        chunked=False a prompt runs whole in one step;
    chunked=True        a step holds at most `token_budget` tokens: decodes
                        first, then prompt chunks that fill what is left (04.3).

Memory runs out: the newest request is preempted, its blocks freed, and it
goes back to the front of the queue to be recomputed (04.4). `max_queue`
turns requests away when too many wait (admission control).

Shortcuts, each with its limit: a request's whole prompt gets blocks when it
is admitted (vLLM allocates as chunks run), a preempted request is always
recomputed (never swapped to CPU), and the simulator prices a step with a
roofline (Module 02) plus a fixed overhead, so it ranks policies well and
predicts absolute speed only as a ceiling.
"""

from dataclasses import dataclass, field
from zlib import crc32

import numpy as np

from .kv import BlockAllocator, OutOfBlocks, PagedView, trace_tokens, step_forward
from .metrics import SLO, Timed, meets, percentile
from .roofline import shape as _shape
from .data import hardware as _hardware, model_config as _model_config
from .engine.sampling import sample

PRIORITY = {"interactive": 0, "agent": 1, "long": 2, "batch": 3}


@dataclass(eq=False)
class Request:
    id: str
    prompt: list                 # token ids
    max_new: int
    slo: str = "interactive"
    arrival: float = 0.0
    # state the scheduler keeps
    tokens: list = field(default_factory=list)   # prompt, then generated tokens
    computed: int = 0            # leading tokens whose keys and values exist
    generated: int = 0
    stamps: list = field(default_factory=list)   # when each new token was made
    preemptions: int = 0
    status: str = "waiting"      # waiting, running, done, rejected
    cached: int = 0              # prompt tokens the prefix cache supplied (first time)

    def __post_init__(self):
        self.tokens = self.prompt[:]

    @property
    def want(self):
        """Tokens still to run before this request can make its next token."""
        return len(self.tokens) - self.computed


def from_trace(row, scale=1.0):
    """A `Request` from a trace row; `scale` replays time that many times faster."""
    return Request(row["id"], trace_tokens(row), row["output_tokens"],
                   row["slo_class"], row["arrival_s"] / scale)


class Scheduler:
    def __init__(self, num_blocks=4096, block_size=16, token_budget=512,
                 chunked=True, max_seqs=64, mode="continuous", prefix_cache=True,
                 policy="fcfs", max_queue=None, alloc=None):
        if mode not in ("static", "continuous"):
            raise ValueError("mode is 'static' or 'continuous'")
        if policy not in ("fcfs", "priority"):
            raise ValueError("policy is 'fcfs' or 'priority'")
        self.alloc = alloc or BlockAllocator(num_blocks, block_size)
        self.token_budget, self.chunked, self.max_seqs = token_budget, chunked, max_seqs
        self.mode, self.prefix_cache, self.policy = mode, prefix_cache, policy
        self.max_queue = max_queue
        self.waiting, self.running = [], []
        self.preemptions = 0

    # -- requests come in --
    def submit(self, r):
        """Queue a request, or turn it away (admission control). Returns True
        if it was queued."""
        if self.max_queue is not None and len(self.waiting) >= self.max_queue:
            r.status = "rejected"
            return False
        self.waiting.append(r)
        return True

    def _next_waiting(self):
        if self.policy == "fcfs":
            return self.waiting[0]
        return min(self.waiting, key=lambda r: (PRIORITY.get(r.slo, 9), r.arrival))

    # -- one step --
    def schedule(self, now=0.0):
        """Choose what the next step runs: a list of (request, tokens)."""
        if self.mode == "static":
            return self._schedule_static()
        budget, plan = self.token_budget, []
        for r in list(self.running):                       # 1. who is already in
            if r.status != "running":
                continue                                    # preempted a moment ago
            if not self._room_for_next_token(r):
                continue
            n = min(r.want, budget) if (self.chunked or r.want == 1) else r.want
            if n > 0:
                plan.append((r, n))
                budget -= n
        while self.waiting and len(self.running) < self.max_seqs:   # 2. who gets in
            r = self._next_waiting()
            if not self.alloc.can_allocate(r.tokens, self.prefix_cache):
                break
            hit = len(self.alloc.match(r.tokens)) * self.alloc.block_size if self.prefix_cache else 0
            want = len(r.tokens) - hit
            if self.chunked:
                n = min(want, budget)
                if n <= 0:
                    break
            else:
                n = want
                if plan and n > budget:
                    break
            self._admit(r)
            plan.append((r, n))
            budget -= n
        return [(r, n) for r, n in plan if r.status == "running"]

    def _admit(self, r):
        self.waiting.remove(r)
        r.cached = self.alloc.allocate(r.id, r.tokens, share=self.prefix_cache)
        r.computed, r.status = r.cached, "running"
        self.running.append(r)

    def _room_for_next_token(self, r):
        """Make sure the block table has a slot for the token about to run;
        if memory is out, preempt (newest running request first)."""
        while self.alloc.seqs[r.id].n_tokens < len(r.tokens):
            try:
                self.alloc.extend(r.id, 1)
            except OutOfBlocks:
                victim = [x for x in self.running if x.status == "running"][-1]
                self.preempt(victim)
                if victim is r:
                    return False
        return True

    def preempt(self, r):
        """Free a running request's blocks and queue it again, first in line.
        Its generated tokens stay, so it re-reads them as prompt (recompute)."""
        self.alloc.free(r.id)
        self.running.remove(r)
        r.computed, r.status = 0, "waiting"
        r.preemptions += 1
        self.preemptions += 1
        self.waiting.insert(0, r)

    def update(self, plan, now, sampled=None):
        """Record a finished step. `now` is when it ended. `sampled` maps a
        request id to the token it made (or None for end of text); without
        it, a made-up token is used, for the simulator."""
        for r, n in plan:
            if r.status != "running":
                continue
            r.computed += n
            self.alloc.commit(r.id, r.tokens, r.computed)
            if r.computed < len(r.tokens):
                continue                                    # still prefilling
            token = (sampled or {}).get(r.id, 10**9 + crc32(r.id.encode()) % 10**6 * 10**3 + r.generated)
            r.stamps.append(now)
            if token is None:
                self._finish(r)
                continue
            r.generated += 1
            r.tokens.append(token)
            if r.generated >= r.max_new:
                self._finish(r)
        if self.mode == "static":
            self._end_of_static_step()

    def _finish(self, r):
        r.status = "done"
        if self.mode == "continuous":
            self.alloc.free(r.id)
            self.running.remove(r)
        # static: a finished request keeps its slot and blocks until the whole batch ends

    # -- static batching (04.1): the baseline the others beat --
    def _schedule_static(self):
        if not self.running:
            while self.waiting and len(self.running) < self.max_seqs:
                r = self._next_waiting()
                if not self.alloc.can_allocate(r.tokens, self.prefix_cache):
                    break
                self._admit(r)
        plan = []
        for r in self.running:
            if r.status == "done":
                plan.append((r, 0))                         # padding: the slot still costs a step
            elif self._room_for_next_token(r):
                plan.append((r, r.want))
        return plan

    def _end_of_static_step(self):
        if self.running and all(r.status == "done" for r in self.running):
            for r in self.running:
                self.alloc.free(r.id)
            self.running.clear()

    @property
    def idle(self):
        return not self.waiting and not self.running


# ---- the cost of a step, for the simulator -----------------------------------

class StepModel:
    """Seconds for one step, from the roofline (Module 02).

    A step reads the weights once, reads the keys and values of every request
    in it, and does 2 multiply-adds per weight per token plus attention. It
    takes as long as the slower of moving the bytes and doing the sums, plus
    `overhead_s` the CPU needs to build the step (lesson 04.7).
    """

    def __init__(self, model="Qwen3-4B", hardware="L4", overhead_s=0.002,
                 weight_bytes_per_param=2.0):
        cfg = _model_config(model) if isinstance(model, str) else model
        chip = _hardware(hardware) if isinstance(hardware, str) else hardware
        s = _shape(cfg)
        self.bw = chip["bandwidth_gbs"] * 1e9
        self.peak = chip["fp16_tflops"] * 1e12
        self.weight_bytes = s["matmul_params"] * weight_bytes_per_param
        self.flops_per_token = 2 * s["matmul_params"]
        self.attn_dim = s["layers"] * s["heads"] * s["head_dim"]
        self.kv_per_token = s["kv_bytes_per_token_f16"]
        self.overhead_s = overhead_s

    def seconds(self, plan):
        """`plan` is [(request, n)]; each request's `computed` is its context
        before the step (padding rows, n = 0, still read their cache)."""
        tokens = sum(n for _, n in plan)
        kv_read = sum(r.computed + n for r, n in plan)       # keys and values read or written
        attn = sum(4 * self.attn_dim * (n * r.computed + n * (n + 1) / 2) for r, n in plan)
        t_mem = (self.weight_bytes + self.kv_per_token * kv_read) / self.bw
        t_cmp = (self.flops_per_token * tokens + attn) / self.peak
        return max(t_mem, t_cmp) + self.overhead_s


def simulate(rows, sched, step_model=None, scale=1.0, max_steps=10_000_000):
    """Replay trace `rows` through `sched` and return the requests.

    Time jumps over idle gaps and advances by `step_model.seconds(plan)` for
    each step. `scale` replays the day that many times faster, to load the
    server. Returns (requests, steps): the `Request` objects (finished,
    rejected or, if max_steps ran out, unfinished) and a list of
    (start_time, tokens, running, waiting) per step.
    """
    step_model = step_model or StepModel()
    rows = sorted(rows, key=lambda r: r["arrival_s"])
    reqs, t, i, steps = [], 0.0, 0, []
    while i < len(rows) or not sched.idle:
        if sched.idle:
            t = max(t, rows[i]["arrival_s"] / scale)
        while i < len(rows) and rows[i]["arrival_s"] / scale <= t:
            reqs.append(from_trace(rows[i], scale))
            sched.submit(reqs[-1])
            i += 1
        plan = sched.schedule(t)
        if not plan and not sched.running and sched.waiting:
            plan = sched.schedule(t)                         # preempted work is admitted now
        if not plan:
            if sched.running or not sched.waiting:
                raise RuntimeError("scheduler returned an empty plan with work left")
            r = sched.waiting.pop(0)                         # can never fit: refuse it
            r.status = "rejected"
            continue
        steps.append((t, sum(n for _, n in plan), len(sched.running), len(sched.waiting)))
        t += step_model.seconds(plan)
        sched.update(plan, t)
        if len(steps) >= max_steps:
            break
    return reqs, steps


def window(rows, start_h, end_h, classes=None):
    """The trace rows that arrive between two hours of the day (IST), in the
    given SLO classes (all if None), with time counted from the first hour."""
    return [{**r, "arrival_s": r["arrival_s"] - start_h * 3600} for r in rows
            if start_h * 3600 <= r["arrival_s"] < end_h * 3600
            and (classes is None or r["slo_class"] in classes)]


def timed(reqs):
    """The finished requests as `metrics.Timed` objects (arrival is `sent`)."""
    return [Timed(r.arrival, r.stamps) for r in reqs if r.status == "done" and r.stamps]


def stalls(reqs):
    """Every gap between a request's consecutive tokens, in seconds: what a
    user watching the stream feels as a freeze."""
    return [g for t in timed(reqs) for g in t.itls]


def report(reqs, steps=()):
    """Headline numbers for a run (seconds, tokens per second, requests per second).

    A request is *good* if it finished and met its class's targets in
    `metrics.SLO` (time to first token, and time per output token on average);
    a request turned away counts as not good. `goodput_rps` is good requests
    per second of the run, `attainment` the good share of everything sent.
    Percentiles are over finished requests.
    """
    done = [(r, Timed(r.arrival, r.stamps)) for r in reqs if r.status == "done" and r.stamps]
    good = sum(meets(t, *SLO[r.slo]) for r, t in done)
    span = max(t.token_times[-1] for _, t in done) - min(r.arrival for r in reqs)
    ttft = [t.ttft for _, t in done]
    gaps = stalls([r for r, _ in done])
    out = {"requests": len(reqs), "finished": len(done),
           "rejected": sum(r.status == "rejected" for r in reqs),
           "preemptions": sum(r.preemptions for r in reqs),
           "attainment": good / len(reqs), "goodput_rps": good / span,
           "tokens_per_s": sum(t.n_tokens for _, t in done) / span,
           "ttft_p50": percentile(ttft, 50), "ttft_p95": percentile(ttft, 95),
           "itl_p99": percentile(gaps, 99) if gaps else None,
           "itl_max": max(gaps) if gaps else None}
    if steps:
        out["mean_batch"] = float(np.mean([s[2] for s in steps]))
    return out


# ---- the real engine, one step at a time (04.8) -------------------------------

class Executor:
    """Runs a plan on the NumPy engine, with keys and values in a `PagedCache`.

        cache = PagedCache(model.config, num_blocks=256)
        sched = Scheduler(alloc=cache.alloc, ...)
        ex = Executor(model, cache)
        plan = sched.schedule(); sched.update(plan, time.time(), ex.run(plan))
    """

    def __init__(self, model, cache, seed=0, **sampling):
        self.model, self.cache, self.sampling = model, cache, sampling
        self.rng = np.random.default_rng(seed)

    def run(self, plan, temperature=0.0):
        """Run every (request, n) together; return {id: next token or None}
        for the requests that finished their tokens this step."""
        items, owners = [], []
        for r, n in plan:
            if n == 0:
                continue
            self.cache.computed[r.id] = r.computed
            items.append((PagedView(self.cache, r.id), r.tokens[r.computed:r.computed + n]))
            owners.append((r, n))
        out = {}
        if not items:
            return out
        logits = step_forward(self.model, items)
        for (r, n), row in zip(owners, logits):
            if r.computed + n < len(r.tokens):
                continue                                     # a prompt chunk: no token yet
            if getattr(r, "ignore_eos", False):
                row = row.copy()
                row[list(self.model.config.eos)] = -np.inf   # a benchmark wants full-length replies
            tok = sample(row, history=r.tokens[len(r.prompt):], rng=self.rng,
                         temperature=temperature, **self.sampling)
            out[r.id] = None if tok in self.model.config.eos else tok
        return out
