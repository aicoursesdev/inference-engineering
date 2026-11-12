"""Embedding throughput against batch size, with a local Ollama.

ollama pull qwen3-embedding:0.6b
uv run --with httpx python embed_throughput.py
Embedding is a prefill-only job: no decode, so the question is how many
tokens per second the model reads, and batching is how you keep it busy.
"""
import time

import httpx

URL = "http://localhost:11434/api/embed"
MODEL = "qwen3-embedding:0.6b"
SENTENCE = ("Tidewater Goods ships the green rain jacket in four sizes and "
            "refunds any order returned within thirty days of delivery. ")


def run(batch, texts_total=256, repeat=6):
    docs = [SENTENCE * repeat + f"Item {i}." for i in range(texts_total)]
    t0 = time.perf_counter()
    tokens = 0
    for i in range(0, texts_total, batch):
        r = httpx.post(URL, json={"model": MODEL, "input": docs[i:i + batch]},
                       timeout=300).json()
        tokens += r["prompt_eval_count"]
    dt = time.perf_counter() - t0
    return texts_total / dt, tokens / dt, tokens


if __name__ == "__main__":
    run(8, 16)  # warm-up: loads the model
    print(f"{'batch':>5s} {'docs/s':>8s} {'tokens/s':>9s}")
    for b in (1, 2, 4, 8, 16, 32, 64):
        d, t, _ = run(b)
        print(f"{b:5d} {d:8.1f} {t:9.0f}")
