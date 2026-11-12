from servinglab.data import traces
from scheduler import Scheduler, StepModel, report, simulate, window


def worst(**kw):
    rows = window(traces("mixed"), 12, 12.5, ["interactive", "long"])
    reqs, steps = simulate(rows, Scheduler(num_blocks=5000, **kw), StepModel())
    return report(reqs, steps)["itl_max"]


def test_a_smaller_budget_never_freezes_streams_longer():
    assert worst(token_budget=256) <= worst(token_budget=2048) <= worst(chunked=False, token_budget=10**9)


def test_a_step_never_holds_more_tokens_than_the_budget():
    from scheduler import Request
    s = Scheduler(num_blocks=256, token_budget=32, max_seqs=4)
    s.submit(Request("long", list(range(100)), 3))
    s.submit(Request("chat", list(range(1000, 1010)), 3))
    while not s.idle:
        plan = s.schedule()
        assert sum(n for _, n in plan) <= 32
        s.update(plan, 0.0)


def test_a_long_prompt_is_split_across_steps_and_decodes_go_first():
    from scheduler import Request
    s = Scheduler(num_blocks=256, token_budget=16, max_seqs=4, prefix_cache=False)
    chat, long = Request("chat", list(range(8)), 5), Request("long", list(range(500, 600)), 2)
    s.submit(chat)
    s.update(s.schedule(), 1.0)                  # chat prefills and makes its first token
    s.submit(long)
    plan = s.schedule()
    assert plan[0] == (chat, 1) and plan[1] == (long, 15)    # decode first, chunk takes the rest
