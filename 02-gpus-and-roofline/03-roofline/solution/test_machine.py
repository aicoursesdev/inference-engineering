import machine


def test_ridge_is_flops_over_bytes():
    assert machine.ridge(989.5e12, 3.35e12) == 989.5 / 3.35  # the H100: about 295


def test_best_of_takes_the_fastest_run():
    # fake clock: three runs of 0.3, 0.1 and 0.2 seconds
    clock = iter(sum(([t, t + d] for t, d in zip([0, 1, 2], [0.3, 0.1, 0.2])), []))
    original = machine.time.perf_counter
    machine.time.perf_counter = lambda: next(clock)
    try:
        assert abs(machine.best_of(lambda: None, repeats=3) - 0.1) < 1e-9
    finally:
        machine.time.perf_counter = original


def test_the_cpu_numbers_are_plausible():
    assert machine.cpu_bandwidth(megabytes=64, threads=2) > 1e9   # more than 1 GB/s on any machine
    assert machine.cpu_flops(n=512) > 1e9                          # more than 1 GFLOPS
