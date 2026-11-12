"""Change one thing: Ollama's context length, same model, same prompt.

Ollama's /api/generate reports its own eval counts and times, so no
client timing is involved.
"""
import statistics

import httpx

from chat_bench import PROMPT


def decode_speed(num_ctx, runs=5):
    rates = []
    for _ in range(runs + 1):  # the first run loads the model; dropped
        r = httpx.post("http://localhost:11434/api/generate", timeout=300, json={
            "model": "qwen3:0.6b", "prompt": PROMPT, "stream": False,
            "think": False,
            "options": {"num_ctx": num_ctx, "num_predict": 128,
                        "temperature": 0}}).json()
        rates.append(r["eval_count"] / (r["eval_duration"] / 1e9))
    return statistics.median(rates[1:])


if __name__ == "__main__":
    for n in (32768, 4096):
        print(f"num_ctx {n:6d}: {decode_speed(n):4.0f} tok/s")
