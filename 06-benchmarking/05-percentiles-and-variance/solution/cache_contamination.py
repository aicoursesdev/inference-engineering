"""The same Kodekart requests with and without a shared prompt start: what a warm prefix cache hides.

    llama-server -m models/Qwen3-0.6B-Q8_0.gguf --port 8080 -c 16384 -np 4
    uv run python cache_contamination.py http://localhost:8080
"""

import asyncio
import sys

from servinglab import bench
from servinglab.data import traces

RATE = 1  # a request a second: the server is never busy, so only the prompt differs


def run(url, share):
    rows = [r for r in traces("kodekart") if r["prefix_id"] == "kk-system"][:30]  # first steps of 30 tasks
    reqs = bench.make_requests(rows, max_input=3500, max_output=32, share_prefix=share)
    results, seconds = asyncio.run(bench.run(url, "local", reqs, RATE))
    return bench.summarize(results, seconds, bench.SLOS["agent"], RATE), results


if __name__ == "__main__":
    url = sys.argv[1]
    run(url, False)  # warm up the server, then throw it away
    print(f"{'prompt starts':<28}{'prompt tokens':>14}{'TTFT p50':>10}{'p95':>8}")
    for name, share in (("all different", False), ("shared 3,000-token system", True)):
        s, results = run(url, share)
        tokens = sum(r["prompt_tokens"] for r in results if r["ok"]) / len(results)
        print(f"{name:<28}{tokens:>14,.0f}{s['ttft']['p50'] * 1000:>8.0f}ms{s['ttft']['p95'] * 1000:>6.0f}ms")
