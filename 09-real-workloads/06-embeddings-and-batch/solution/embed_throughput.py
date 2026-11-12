"""Embedding throughput against batch size: one request at a time, or many
texts in one request, with Qwen3-Embedding-0.6B (Q8_0).

    llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 --embeddings \
        -c 4096 -np 1 --port 8080
    uv run python embed_throughput.py [http://localhost:8080]
"""

import sys
import time

import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
LONG = ("Order 4471: one Tidewater rain jacket, size M, returned unworn within "
        "ten days of delivery; refund to the original payment method. ") * 4
SHORT = "refund for order 4471"      # a search query is this short
TOTAL = 64  # texts to embed in every run


def run(batch, text):
    t = time.perf_counter()
    tokens = 0
    for _ in range(TOTAL // batch):
        r = httpx.post(URL + "/v1/embeddings", timeout=300,
                       json={"model": "m", "input": [text] * batch}).json()
        tokens += r["usage"]["prompt_tokens"]
    return tokens, time.perf_counter() - t


run(1, SHORT)  # warm up
print(f"{'text':>6} {'texts per request':>18} {'tokens':>7} {'seconds':>8} {'tokens/s':>9}")
for name, text in (("short", SHORT), ("long", LONG)):
    for batch in (1, 4, 16, 64):
        best = min((run(batch, text) for _ in range(3)), key=lambda r: r[1])
        print(f"{name:>6} {batch:18d} {best[0]:7d} {best[1]:8.2f} "
              f"{best[0] / best[1]:9.0f}")
