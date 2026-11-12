"""Split every request's time into queue, prefill and decode, from a recorded run.

    uv run python split_request.py good      # or: bad

Reads data/recorded/incidents/incident1-<run>.jsonl (made by 12.7's recorder against a real
llama-server). Each row has the client's clock (ttft_ms, total_ms) and the server's own
timings (prompt_ms, predicted_ms). The server does not report the wait for a free slot,
so "queue and network" is what is left: ttft_ms minus the server's prefill time.
"""
import json, statistics, sys
from pathlib import Path

run = sys.argv[1] if len(sys.argv) > 1 else "good"
path = Path(__file__).resolve().parents[3] / "data" / "recorded" / "incidents" / f"incident1-{run}.jsonl"
rows = [json.loads(l) for l in path.read_text().splitlines()]
for r in rows:
    r["wait_ms"] = r["ttft_ms"] - r["prompt_ms"]          # waiting before the first token that is not prefill
med = lambda k: statistics.median(r[k] for r in rows)
print(f"{run}: {len(rows)} requests")
print(f"  queue and network  median {med('wait_ms'):7.1f} ms")
print(f"  prefill            median {med('prompt_ms'):7.1f} ms   ({med('prompt_n'):.0f} prompt tokens computed, {med('cache_n'):.0f} reused)")
print(f"  decode             median {med('predicted_ms'):7.1f} ms   ({med('predicted_n'):.0f} tokens)")
slow = max(rows, key=lambda r: r["ttft_ms"])
print(f"  slowest first token: agent {slow['agent']} step {slow['step']}: {slow['ttft_ms']:.0f} ms = "
      f"{slow['wait_ms']:.0f} waiting + {slow['prompt_ms']:.0f} prefill")
