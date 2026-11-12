"""Send the same chat request several ways and print the four numbers.

    llama-server -hf Qwen/Qwen3-0.6B-GGUF:Q8_0 --port 8081
    uv run python measure_load.py
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from servinglab.metrics import SLO, summary, timed_stream

URL = f"http://127.0.0.1:{os.environ.get('PORT', '8081')}/v1/chat/completions"
BODY = {
    "model": "m", "temperature": 0, "max_tokens": 120,
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [{"role": "user", "content":
                  "Explain in four sentences how to dry a wet rain jacket."}],
}


def run(n_requests, in_flight):
    """n_requests in total, never more than in_flight at once."""
    with ThreadPoolExecutor(in_flight) as pool:
        return list(pool.map(lambda _: timed_stream(URL, BODY),
                             range(n_requests)))


run(2, 1)  # warm up
saved = {}
print("in flight  TTFT p95   TPOT p50  tokens/s  good/s  good share")
for in_flight in (1, 4, 8, 16):
    reqs = run(32, in_flight)
    saved[in_flight] = [[r.sent, r.token_times] for r in reqs]
    s = summary(reqs, SLO["interactive"])
    print(f"{in_flight:9d}  {s['ttft_p95'] * 1000:7.0f} ms  "
          f"{s['tpot_p50'] * 1000:6.1f} ms  {s['tokens_per_s']:8.0f}  "
          f"{s['goodput_rps']:6.1f}  {s['good_fraction']:9.0%}")

if len(sys.argv) > 1:  # python measure_load.py out.json keeps every timing
    json.dump(saved, open(sys.argv[1], "w"))
