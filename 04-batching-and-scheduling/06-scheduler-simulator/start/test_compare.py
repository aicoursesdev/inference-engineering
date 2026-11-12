from servinglab.data import traces
from scheduler import Scheduler, StepModel, report, simulate, window


def goodput(**kw):
    rows = window(traces("mixed"), 10, 11, ["interactive", "long"])
    reqs, steps = simulate(rows, Scheduler(num_blocks=5000, max_seqs=32, **kw), StepModel(), scale=60)
    return report(reqs, steps)["tokens_per_s"]


def test_continuous_out_produces_static_under_load():
    assert goodput(chunked=False, token_budget=10**9) > goodput(mode="static")


def test_a_lone_step_costs_what_module_02_predicts():
    from servinglab.roofline import predict
    from scheduler import Request
    sm = StepModel("Qwen3-4B", "L4", overhead_s=0)
    r = Request("a", list(range(512)), 4)
    r.computed = 512
    assert abs(sm.seconds([(r, 1)]) / predict("Qwen3-4B", "L4", batch=1, context=512)["decode"]["seconds"] - 1) < 1e-3
    r = Request("b", list(range(2000)), 4)
    assert abs(sm.seconds([(r, 2000)]) / predict("Qwen3-4B", "L4", prompt=2000)["prefill"]["seconds"] - 1) < 1e-3
