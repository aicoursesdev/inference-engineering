"""A fair timing and reporting harness for a kernel PR. The statistics are
plain Python (tested offline); `timeit` needs a CUDA GPU and PyTorch."""
import statistics


def summarize(times_ms):
    """Median and the spread a reviewer wants: p10 and p90."""
    s = sorted(times_ms)
    q = statistics.quantiles(s, n=10, method="inclusive")   # 9 cut points
    return {"median": statistics.median(s), "p10": q[0], "p90": q[-1],
            "n": len(s)}


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
    base = rows[0]["median"]
    out = ["| kernel | median ms | p10 | p90 | vs first row |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['name']} | {r['median']:.4f} | {r['p10']:.4f} | "
                   f"{r['p90']:.4f} | {base / r['median']:.2f}x |")
    return "\n".join(out)
