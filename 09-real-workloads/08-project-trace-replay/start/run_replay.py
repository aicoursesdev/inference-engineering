"""The project: replay Kodekart's agent traffic against a local server, with
and without prefix reuse and with a JSON-schema constraint, and price it.

    llama-server -m Qwen3-0.6B-Q8_0.gguf -c 4096 -np 2 --port 8080
    uv run python run_replay.py [http://localhost:8080] [tasks] [scale]

Lengths are scaled down (0.1 by default) because a laptop cannot read
15,000 tokens a step; the shape (a system prompt, then a prompt that grows
by a tool result each step) is the trace's.
"""

import asyncio
import json
import sys

from servinglab.data import prices
from replay import Prices, build_requests, replay, report, trace_hit_rate

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
TASKS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
SCHEMA = {"type": "object", "required": ["tool", "line"],
          "properties": {"tool": {"enum": ["read_file", "edit_file", "run_tests"]},
                         "line": {"type": "integer"}}}

# Monsoon prices a million tokens like DeepInfra's Qwen3-14B list price
# (data/prices.json); a cache hit is charged at a quarter of that, the ratio
# DeepSeek published for its disk cache (R1, February 2025). Both are inputs.
qwen = next(p for p in prices()["api_per_million_tokens"] if p["model"] == "Qwen3-14B")
PRICE = Prices(input=qwen["input_usd_per_mtok"], output=qwen["output_usd_per_mtok"],
               cached_input=qwen["input_usd_per_mtok"] * 0.25)

reqs = build_requests("kodekart", scale=SCALE, tasks=TASKS, max_tokens=64)
print(f"{len(reqs)} requests in {TASKS} tasks; the trace allows "
      f"{trace_hit_rate(reqs):.0%} reuse")
asyncio.run(replay(reqs[:2], URL, "m"))      # warm up the server

print(f"{'run':20} {'hit rate':>9} {'ttft p50':>9} {'cost/task':>11} {'valid':>6}")
for label, cache, schema in (("no reuse", False, None), ("prefix reuse", True, None),
                             ("reuse + schema", True, SCHEMA)):
    rows = asyncio.run(replay(reqs, URL, "m", concurrency=2, cache=cache,
                              schema=schema))
    rep = report(rows, PRICE)
    ok = [r for r in rows if r.valid_json is not None]
    valid = f"{sum(r.valid_json for r in ok)}/{len(ok)}" if ok else "-"
    print(f"{label:20} {rep['hit_rate']:9.0%} {rep['ttft_p50_s'] * 1000:7.0f}ms "
          f"${rep['cost_per_task_usd']:.6f} {valid:>6}"
          + (f"  ({rep['errors']} errors)" if rep["errors"] else ""))
    json.dump(rep, open(f"replay_{label.replace(' + ', '_').replace(' ', '_')}.json", "w"))
