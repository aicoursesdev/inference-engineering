"""A fair timing and reporting harness for a kernel PR. The statistics are
plain Python (tested offline); `timeit` needs a CUDA GPU and PyTorch."""
import statistics


def summarize(times_ms):
    """Median and the spread a reviewer wants: p10 and p90."""
    raise NotImplementedError("TODO 16.7: median plus p10 and p90 (statistics.quantiles, n=10, method=inclusive) and n, see the lesson's 'Benchmark against the engine' section")


def timeit(fn, warmup=10, reps=50, flush_l2=True):
    """Per-call milliseconds with CUDA events. Flushing L2 between calls keeps
    a kernel from being timed on weights a previous call left in cache."""
    import torch
    scratch = torch.empty(256 * 2**20 // 4, dtype=torch.float32, device="cuda")
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        if flush_l2:
            scratch.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return times


def markdown_table(rows):
    """rows: [{"name", "median", "p10", "p90"}] -> a table for the PR text."""
    raise NotImplementedError("TODO 16.7: one row per kernel with median, p10, p90 and the first row's median divided by this median, see the lesson's 'Benchmark against the engine' section")
