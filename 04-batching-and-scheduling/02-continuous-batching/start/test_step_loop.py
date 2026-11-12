from scheduler import Request, Scheduler


def steps(mode, lengths=(5, 40, 5, 5), slots=2):
    s = Scheduler(num_blocks=128, max_seqs=slots, mode=mode, token_budget=4096)
    for i, n in enumerate(lengths):
        s.submit(Request(f"r{i}", [i * 10 + j for j in range(4)], n))
    k = 0
    while not s.idle:
        plan = s.schedule(); k += 1; s.update(plan, float(k))
    return k


def test_continuous_never_takes_more_steps_than_static():
    assert steps("continuous") < steps("static")
