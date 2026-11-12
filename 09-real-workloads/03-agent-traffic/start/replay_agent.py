"""Replay four Kodekart agent tasks, with and without prefix reuse, and read the
hit rate from the server.

    llama-server -m Qwen3-0.6B-Q8_0.gguf -c 4096 -np 2 --port 8080
    uv run python replay_agent.py [http://localhost:8080]

Lengths are scaled to a tenth: a laptop cannot read 15,000-token prompts a step.
"""

import asyncio
import sys

from servinglab.data import prices
from servinglab.replay import Prices, build_requests, replay, report, trace_hit_rate

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
qwen = next(p for p in prices()["api_per_million_tokens"] if p["model"] == "Qwen3-14B")
# a cache hit costs a quarter of a miss, the ratio of DeepSeek's R1 list (0.14 vs 0.55)
PRICE = Prices(input=qwen["input_usd_per_mtok"], output=qwen["output_usd_per_mtok"],
               cached_input=qwen["input_usd_per_mtok"] * 0.25)

reqs = build_requests("kodekart", scale=0.1, tasks=4, max_tokens=64)
print(f"{len(reqs)} requests in 4 tasks; the trace allows "
      f"{trace_hit_rate(reqs):.0%} reuse")
asyncio.run(replay(reqs[:2], URL, "m"))          # warm up the server
print(f"{'run':14} {'hit rate':>9} {'ttft p50':>9} {'cost/task':>11}")
for label, cache in (("no reuse", False), ("prefix reuse", True)):
    rep = report(asyncio.run(replay(reqs, URL, "m", concurrency=2, cache=cache)), PRICE)
    print(f"{label:14} {rep['hit_rate']:9.0%} {rep['ttft_p50_s'] * 1000:7.0f}ms "
          f"${rep['cost_per_task_usd']:.6f}")
