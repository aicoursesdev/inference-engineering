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
    raise NotImplementedError(
        "TODO 06.6: return the send time of each request: gamma-distributed gaps with mean 1/rate (shape = burstiness), the first request at time 0, see lesson 06.2's 'Arrivals: when requests come' section")


def meets(res, slo):
    """Did this request succeed with TTFT and TPOT inside the target?"""
    raise NotImplementedError(
        "TODO 06.6: true only if the request succeeded and met both the TTFT and the TPOT target, see lesson 06.4's 'Goodput and capacity' section")


def summarize(results, seconds, slo, offered_rate=None):
    """Percentiles, throughput and goodput of one run (`seconds` is its wall time).

    Return the dict the saved sweep holds under each run's "summary" (look at
    data/recorded/bench-sweep-tidewater-qwen3-0.6b.json for its keys)."""
    raise NotImplementedError(
        "TODO 06.6: one run's percentiles, throughput, goodput, SLO attainment and max lag, see lesson 06.5's 'Percentiles, and how many requests they need' section")


def capacity(summaries, attainment=0.9):
    """The highest offered rate where `attainment` of requests met the SLO,
    or None if even the lowest rate failed."""
    raise NotImplementedError(
        "TODO 06.6: the highest offered rate where 90 percent of requests met the SLO, or None, see lesson 06.4's 'Goodput and capacity' section")


def report(data):
    """The sweep as a markdown table, with its conditions first and the
    capacity line last."""
    raise NotImplementedError(
        "TODO 06.6: the markdown table of the sweep with its conditions first and the capacity line last, see lesson 06.4's 'Sweep the rate' section")


def main():
    # bench.sweep and bench.main look these names up in bench, so this runs your code.
    bench.arrival_times, bench.meets, bench.summarize = arrival_times, meets, summarize
    bench.capacity, bench.report = capacity, report
    bench.main()


if __name__ == "__main__":
    main()
