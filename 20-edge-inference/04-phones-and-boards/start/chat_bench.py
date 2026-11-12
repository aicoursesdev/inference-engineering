"""Measure one OpenAI-compatible chat endpoint the same way every time.

python chat_bench.py http://localhost:8093 any-name
Reports time to first token and decode speed (tokens after the first,
divided by the time they took), median of N runs, same prompt, greedy.
"""
import json
import statistics
import sys
import time

import httpx

PROMPT = ("Explain in about two hundred words how a rain jacket keeps "
          "water out while letting sweat escape.")


def one(base, model, max_tokens=128):
    body = {"model": model, "stream": True, "temperature": 0,
            "max_tokens": max_tokens, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": PROMPT}]}
    t0, first, last, chunks, usage = time.perf_counter(), None, None, 0, None
    with httpx.stream("POST", base + "/v1/chat/completions", json=body,
                      timeout=300) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            c = json.loads(line[6:])
            usage = c.get("usage") or usage
            ch = c["choices"][0]["delta"] if c.get("choices") else {}
            if any(ch.get(k) for k in ("content", "reasoning", "reasoning_content")):
                now = time.perf_counter()
                first = first or now
                last, chunks = now, chunks + 1
    n = (usage or {}).get("completion_tokens", chunks)
    return first - t0, (n - 1) / (last - first), n


def bench(base, model, runs=5):
    one(base, model, 8)  # warm-up: loads the model, fills caches
    rs = [one(base, model) for _ in range(runs)]
    return (statistics.median(r[0] for r in rs),
            statistics.median(r[1] for r in rs), rs[0][2])


if __name__ == "__main__":
    ttft, tps, n = bench(sys.argv[1], sys.argv[2])
    print(f"TTFT {ttft * 1000:.0f} ms   decode {tps:.0f} tok/s   ({n} tokens)")
