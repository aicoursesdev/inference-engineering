from servinglab.data import traces
from scheduler import Scheduler, StepModel, report, simulate, window


def test_a_tiny_pool_preempts_but_still_finishes_everything():
    batch = window(traces("mixed"), 2, 3, ["batch"])[:120]
    reqs, steps = simulate(batch, Scheduler(num_blocks=300, max_seqs=64), StepModel())
    r = report(reqs, steps)
    assert r["preemptions"] > 0 and r["finished"] == 120


def test_a_queue_cap_turns_some_away_and_helps_the_rest():
    chat = window(traces("mixed"), 9, 18, ["interactive"])[:400]
    open_ = report(*simulate(chat, Scheduler(num_blocks=5000), StepModel(), scale=300))
    capped = report(*simulate(chat, Scheduler(num_blocks=5000, max_queue=10), StepModel(), scale=300))
    assert capped["rejected"] > 0 and capped["ttft_p95"] < open_["ttft_p95"]


def test_priority_lets_chat_overtake_batch_and_fcfs_does_not():
    from scheduler import Request
    for policy, first in (("fcfs", "job"), ("priority", "chat")):
        s = Scheduler(num_blocks=256, policy=policy)
        s.submit(Request("job", [1, 2, 3], 2, slo="batch", arrival=0.0))
        s.submit(Request("chat", [4, 5, 6], 2, slo="interactive", arrival=1.0))
        assert s._next_waiting().id == first


def test_a_full_queue_refuses_and_marks_the_request_rejected():
    from scheduler import Request
    s = Scheduler(num_blocks=256, max_queue=1)
    a, b = Request("a", [1, 2], 1), Request("b", [3, 4], 1)
    assert s.submit(a) is True and s.submit(b) is False and b.status == "rejected"


def test_preempting_frees_blocks_and_requeues_first():
    from scheduler import Request
    s = Scheduler(num_blocks=8, block_size=4, max_seqs=2, prefix_cache=False)
    a, b = Request("a", list(range(8)), 5), Request("b", list(range(10, 14)), 5)
    s.submit(a); s.submit(b)
    s.update(s.schedule(), 1.0)
    free = s.alloc.free_blocks
    s.preempt(b)
    assert s.alloc.free_blocks > free and s.waiting[0] is b and b.computed == 0
    assert b.preemptions == 1
