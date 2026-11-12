"""Embedding throughput against batch size on llama-server.

llama-server -m <qwen3-embedding GGUF> --embeddings --pooling last \
  -b 4096 -ub 4096 --port 8092
uv run --with httpx python embed_llamacpp.py
"""
import time

import httpx

from embed_throughput import SENTENCE

URL = "http://localhost:8092/v1/embeddings"


def run(batch, texts_total=256, repeat=6):
    docs = [SENTENCE * repeat + f"Item {i}." for i in range(texts_total)]
    t0, tokens = time.perf_counter(), 0
    for i in range(0, texts_total, batch):
        r = httpx.post(URL, json={"input": docs[i:i + batch]}, timeout=300)
        tokens += r.json()["usage"]["prompt_tokens"]
    dt = time.perf_counter() - t0
    return texts_total / dt, tokens / dt


if __name__ == "__main__":
    run(8, 16)
    print(f"{'batch':>5s} {'docs/s':>8s} {'tokens/s':>9s}")
    for b in (1, 2, 4, 8, 16, 32, 64):
        d, t = run(b)
        print(f"{b:5d} {d:8.1f} {t:9.0f}")
