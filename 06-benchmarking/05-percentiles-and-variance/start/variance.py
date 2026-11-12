"""Run one benchmark point six times and see how much the answer moves.

    llama-server -m models/Qwen3-0.6B-Q8_0.gguf --port 8080 -c 16384 -np 4
    uv run python variance.py http://localhost:8080
"""

import asyncio
import random
import sys

from servinglab import bench
from servinglab.data import traces

RATE, N = 5, 100


def one_run(url, seed):
    rows = traces("tidewater")
    random.Random(seed).shuffle(rows)
    reqs = bench.make_requests(rows[:N], max_input=1500, max_output=128, share_prefix=True, seed=seed)
    results, seconds = asyncio.run(bench.run(url, "local", reqs, RATE, seed=seed))
    return bench.summarize(results, seconds, bench.SLOS["interactive"], RATE)


if __name__ == "__main__":
    url = sys.argv[1]
    one_run(url, 99)  # warm up, discard
    print(f"{'seed':>5}{'TTFT p50':>10}{'p95':>8}{'p99':>8}{'TPOT p95':>10}{'met SLO':>9}")
    for seed in (0, 1, 2, 3, 4, 0):  # the last run repeats the first exactly
        s = one_run(url, seed)
        print(f"{seed:>5}{s['ttft']['p50'] * 1000:>8.0f}ms{s['ttft']['p95'] * 1000:>6.0f}ms{s['ttft']['p99'] * 1000:>6.0f}ms"
              f"{s['tpot']['p95'] * 1000:>8.0f}ms{s['slo_attainment']:>9.0%}")
