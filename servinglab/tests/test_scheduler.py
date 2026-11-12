"""The scheduler and its simulator (Module 04). Offline, no model."""

import json
from pathlib import Path

import numpy as np

from servinglab.data import traces
from servinglab.engine import Config, Model, generate
from servinglab.engine.weights import read_safetensors
from servinglab.kv import PagedCache
from servinglab.scheduler import (Executor, Request, Scheduler, StepModel,
                                  report, simulate, stalls)

TINY = Path(__file__).parent / "data" / "tiny-llama"


def req(i, n_in, n_out, slo="interactive", t=0.0):
    return Request(f"r{i}", list(range(1000 * i, 1000 * i + n_in)), n_out, slo, t)


def run(sched, requests, now=0.0):
    """Step until idle with unit-time steps; return the steps' plans."""
    for r in requests:
        sched.submit(r)
    plans = []
    while not sched.idle:
        plan = sched.schedule(now)
        assert plan, "stuck"
        plans.append([(r.id, n) for r, n in plan])
        now += 1
        sched.update(plan, now)
    return plans


def test_every_request_gets_all_its_tokens():
    s = Scheduler(num_blocks=64, token_budget=64)
    rs = [req(i, 20, 5) for i in range(3)]
    run(s, rs)
    assert all(r.status == "done" and r.generated == 5 and len(r.stamps) == 5 for r in rs)
    assert s.alloc.free_blocks == 64


def test_a_step_never_exceeds_the_token_budget_when_chunked():
    s = Scheduler(num_blocks=256, token_budget=32, chunked=True)
    plans = run(s, [req(i, 100, 8) for i in range(4)])
    assert all(sum(n for _, n in p) <= 32 for p in plans)


def test_chunking_splits_a_long_prompt_and_decodes_keep_running():
    s = Scheduler(num_blocks=256, token_budget=16, chunked=True)
    short, long_ = req(0, 4, 20), req(1, 100, 2)
    s.submit(short)
    for _ in range(3):                      # let the short one start decoding
        plan = s.schedule(0)
        s.update(plan, 1)
    s.submit(long_)
    plan = dict((r.id, n) for r, n in s.schedule(0))
    assert plan["r0"] == 1 and 0 < plan["r1"] <= 15


def test_unchunked_prompt_runs_whole_in_one_step():
    s = Scheduler(num_blocks=256, token_budget=10**9, chunked=False)
    plans = run(s, [req(0, 100, 3)])
    assert plans[0] == [("r0", 100)]


def test_a_finished_request_frees_its_slot_at_once_in_continuous_mode():
    s = Scheduler(num_blocks=256, max_seqs=2, token_budget=64)
    a, b, c = req(0, 8, 2), req(1, 8, 30), req(2, 8, 2)
    plans = run(s, [a, b, c])
    first_c = next(i for i, p in enumerate(plans) if "r2" in [x for x, _ in p])
    last_b = max(i for i, p in enumerate(plans) if "r1" in [x for x, _ in p])
    assert first_c < last_b                  # c joined while b was still going


def test_static_batching_makes_a_new_request_wait_for_the_slowest():
    s = Scheduler(num_blocks=256, max_seqs=2, mode="static", token_budget=64)
    a, b, c = req(0, 8, 2), req(1, 8, 30), req(2, 8, 2)
    plans = run(s, [a, b, c])
    first_c = next(i for i, p in enumerate(plans) if "r2" in [x for x, _ in p])
    last_b = max(i for i, p in enumerate(plans) if "r1" in [x for x, _ in p])
    assert first_c > last_b


def test_memory_pressure_preempts_the_newest_and_everyone_still_finishes():
    s = Scheduler(num_blocks=12, block_size=4, token_budget=64, prefix_cache=False)
    rs = [req(i, 12, 20) for i in range(3)]      # each needs 8 blocks at the end; 12 in all
    run(s, rs)
    assert s.preemptions > 0
    assert all(r.status == "done" and r.generated == 20 for r in rs)
    assert s.alloc.free_blocks == 12


def test_admission_control_turns_requests_away():
    s = Scheduler(num_blocks=64, max_queue=2)
    rs = [req(i, 8, 2) for i in range(5)]
    assert [s.submit(r) for r in rs] == [True, True, False, False, False]
    assert rs[4].status == "rejected"


def test_priority_policy_lets_chat_overtake_batch():
    s = Scheduler(num_blocks=64, max_seqs=1, policy="priority")
    s.submit(req(0, 8, 2, "batch", 0.0))
    s.submit(req(1, 8, 2, "interactive", 1.0))
    assert s.schedule(0)[0][0].id == "r1"


def test_prefix_cache_skips_shared_prompt_tokens():
    s = Scheduler(num_blocks=64, block_size=4, token_budget=64)
    shared = list(range(40))
    a = Request("a", shared + [1000], 2)
    b = Request("b", shared + [2000], 2)
    run(s, [a])
    run(s, [b])
    assert b.cached == 40 and a.cached == 0


def test_simulation_is_deterministic_and_finishes_a_trace_slice():
    rows = traces("tidewater")[:80]
    outs = []
    for _ in range(2):
        s = Scheduler(num_blocks=3000)
        reqs, steps = simulate(rows, s, StepModel(), scale=20)
        outs.append(report(reqs, steps))
    assert outs[0] == outs[1]
    assert outs[0]["finished"] == 80 and outs[0]["rejected"] == 0


def test_chunked_prefill_shortens_the_worst_stall():
    rows = [r for r in traces("mixed") if r["slo_class"] in ("interactive", "long")][:400]
    worst = {}
    for name, kw in (("whole", dict(chunked=False, token_budget=10**9)),
                     ("chunked", dict(chunked=True, token_budget=512))):
        s = Scheduler(num_blocks=5000, **kw)
        reqs, _ = simulate(rows, s, StepModel(), scale=60)
        worst[name] = max(stalls(reqs))
    assert worst["chunked"] < worst["whole"]


def test_real_engine_batch_matches_one_at_a_time():
    config = Config.from_json(json.loads((TINY / "config.json").read_text()))
    m = Model(config, read_safetensors(TINY / "model.safetensors"), None, TINY)
    m.config = Config(**{**m.config.__dict__, "eos": (-1,)})
    prompts = [[5, 17, 3, 99, 42], [7, 61, 2, 90, 14, 33, 8, 21, 50]]
    cache = PagedCache(m.config, 64, 4)
    s = Scheduler(alloc=cache.alloc, token_budget=6, chunked=True)
    ex = Executor(m, cache)
    rs = [Request(f"q{i}", p, 8) for i, p in enumerate(prompts)]
    for r in rs:
        s.submit(r)
    while not s.idle:
        plan = s.schedule()
        s.update(plan, 0.0, ex.run(plan))
    for r, p in zip(rs, prompts):
        assert r.tokens[len(p):] == list(generate(m, p, 8, temperature=0))
