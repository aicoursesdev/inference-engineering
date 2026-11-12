"""Your benchmark harness: the measuring parts, run through the real sweep.

    uv run python harness.py --url http://localhost:8080 --model local \
        --trace tidewater --rates 1,2,4,6,8,12,16 --requests 100 \
        --slo interactive --engine "llama.cpp, Qwen3-0.6B-Q8_0" --out sweep.json
    uv run python harness.py --report sweep.json

The parts you write are the five that decide what a number means: when
requests are sent, whether one met the target, the summary of a run, the
capacity of a sweep and the report. The streamed request itself, the sweep
loop and the command line come from servinglab.bench; `main()` below swaps
your five in for the finished ones, so the sweep runs on your code.
"""

import numpy as np

from servinglab import bench


def arrival_times(rate, n, burstiness=1.0, seed=0):
    """When each of n requests is sent, in seconds from the start. Gaps are
    gamma-distributed with mean 1/rate (burstiness 1 is a Poisson process,
    below 1 burstier); the first request goes out at time 0."""
    rng = np.random.default_rng(seed)
    gaps = rng.gamma(shape=burstiness, scale=1 / (rate * burstiness), size=n)
    return np.cumsum(gaps) - gaps[0]


def meets(res, slo):
    """Did this request succeed with TTFT and TPOT inside the target?"""
    return res["ok"] and res["ttft"] <= slo["ttft"] and (res["tpot"] or 0) <= slo["tpot"]


def summarize(results, seconds, slo, offered_rate=None):
    """Percentiles, throughput and goodput of one run (`seconds` is its wall time)."""
    ok = [r for r in results if r["ok"]]
    good = sum(meets(r, slo) for r in results)
    s = {"offered_rate": offered_rate, "requests": len(results),
         "errors": len(results) - len(ok), "seconds": seconds, "slo": slo,
         "request_throughput": len(ok) / seconds,
         "output_tokens_per_s": sum(r["output_tokens"] for r in ok) / seconds,
         "goodput": good / seconds, "slo_attainment": good / len(results),
         "max_lag": max((r["lag"] for r in results), default=0)}
    for key in ("ttft", "tpot", "e2e"):
        vals = [r[key] for r in ok if r.get(key) is not None]
        s[key] = {f"p{p}": bench.percentile(vals, p) for p in (50, 95, 99)}
        s[key]["mean"] = float(np.mean(vals)) if vals else None
    return s


def capacity(summaries, attainment=0.9):
    """The highest offered rate where `attainment` of requests met the SLO,
    or None if even the lowest rate failed."""
    good = [s["offered_rate"] for s in summaries if s["slo_attainment"] >= attainment]
    return max(good) if good else None


def report(data):
    """The sweep as a markdown table, with its conditions first and the
    capacity line last."""
    slo = data["slo"]
    lines = [f"# {data.get('label') or 'Benchmark'}",
             f"server {data['server']}, model {data['model']}, {data['requests_per_run']} requests per rate, "
             f"burstiness {data['burstiness']}, seed {data['seed']}, {data['created']}",
             f"engine: {data['engine'] or 'not stated'}; machine: {data['machine']}",
             f"SLO: TTFT <= {slo['ttft']} s and TPOT <= {slo['tpot'] * 1000:.0f} ms", "",
             "| rate (req/s) | done/s | tok/s | TTFT p50 | TTFT p95 | TTFT p99 | TPOT p95 (ms) | met SLO | goodput/s |",
             "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, k=1: "-" if v is None else f"{v * k:.2f}"
    for r in data["runs"]:
        s = r["summary"]
        lines.append(f"| {s['offered_rate']:g} | {s['request_throughput']:.2f} | {s['output_tokens_per_s']:.0f} | "
                     f"{fmt(s['ttft']['p50'])} | {fmt(s['ttft']['p95'])} | {fmt(s['ttft']['p99'])} | "
                     f"{fmt(s['tpot']['p95'], 1000)} | {s['slo_attainment']:.0%} | {s['goodput']:.2f} |")
    cap = data["capacity_rate"]
    lines += ["", f"Capacity: {f'{cap:g} requests per second' if cap else 'none: the lowest rate already missed the SLO'}"
              " (the highest rate where 90% of requests met the SLO)."]
    return "\n".join(lines)


def main():
    # bench.sweep and bench.main look these names up in bench, so this runs your code.
    bench.arrival_times, bench.meets, bench.summarize = arrival_times, meets, summarize
    bench.capacity, bench.report = capacity, report
    bench.main()


if __name__ == "__main__":
    main()
