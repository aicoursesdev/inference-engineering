from servinglab.data import traces
from scheduler import Scheduler, StepModel, report, simulate, window


def tok_s(**kw):
    rows = window(traces("mixed"), 2, 3, ["batch"])[:200]
    return report(*simulate(rows, Scheduler(num_blocks=5000, max_seqs=16, **kw), StepModel()))["tokens_per_s"]


def test_continuous_batching_moves_more_tokens_than_static():
    assert tok_s(chunked=False, token_budget=10**9) > 1.3 * tok_s(mode="static")
