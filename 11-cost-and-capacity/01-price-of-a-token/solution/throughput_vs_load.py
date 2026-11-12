"""Tokens per second against how many requests are in flight, on one llama-server.

    llama-server -m models/Qwen3-4B-Q4_K_M.gguf --port 8081 -np 16 -c 16384
    uv run python throughput_vs_load.py
"""
import os
from concurrent.futures import ThreadPoolExecutor

from servinglab.metrics import SLO, summary, timed_stream

URL = f"http://127.0.0.1:{os.environ.get('PORT', '8081')}/v1/chat/completions"
BODY = {"model": "m", "temperature": 0, "max_tokens": 120,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content":
                      "Explain in four sentences how to dry a wet rain jacket."}]}


def run(n_requests, in_flight):
    with ThreadPoolExecutor(in_flight) as pool:
        return list(pool.map(lambda _: timed_stream(URL, BODY), range(n_requests)))


run(2, 1)  # warm up
print("in flight  tokens/s  TPOT p50   TTFT p95")
for in_flight in (1, 2, 4, 8, 16):
    s = summary(run(4 * in_flight if in_flight > 2 else 8, in_flight), SLO["interactive"])
    print(f"{in_flight:9d}  {s['tokens_per_s']:8.0f}  {s['tpot_p50'] * 1000:6.1f} ms  "
          f"{s['ttft_p95'] * 1000:6.0f} ms")
