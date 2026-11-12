"""Time one streamed reply from any OpenAI-compatible server.

    uv run python measure_stream.py http://127.0.0.1:8080/v1 MODEL [runs]

Prints, per run: time to first token (ms), decode speed (tokens/s, from the
first token to the last) and the token count the server reports.
"""
import json
import statistics
import sys
import time

import httpx

PROMPT = ("Write a long, friendly reply of about 200 words to a customer "
          "in Kochi who asks whether our rain jacket keeps a laptop dry "
          "on a motorbike. Mention the zip, the seams and the hood.")


def one_run(base, model, max_tokens=128):
    body = {"model": model, "stream": True, "temperature": 0,
            "max_tokens": max_tokens,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": PROMPT + " /no_think"}]}
    t0 = time.perf_counter()
    first = last = None
    tokens = 0
    with httpx.stream("POST", base + "/chat/completions", json=body,
                      timeout=120) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            if chunk["choices"] and chunk["choices"][0]["delta"].get("content"):
                last = time.perf_counter()
                first = first or last
    ttft = (first - t0) * 1000
    tps = (tokens - 1) / (last - first)
    return ttft, tps, tokens


if __name__ == "__main__":
    base, model = sys.argv[1], sys.argv[2]
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    one_run(base, model)  # warm-up, not counted
    rows = [one_run(base, model) for _ in range(runs)]
    for ttft, tps, n in rows:
        print(f"ttft {ttft:7.1f} ms   decode {tps:6.1f} tok/s   {n} tokens")
    print(f"median ttft {statistics.median(r[0] for r in rows):.1f} ms, "
          f"median decode {statistics.median(r[1] for r in rows):.1f} tok/s")
