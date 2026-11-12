"""Scrape llama-server's /metrics once a second while eight clients keep it busy.

    llama-server -m models/Qwen3-0.6B-Q8_0.gguf --port 8081 -np 2 --metrics
    uv run python scrape.py
"""
import re
import threading
import time

import httpx
from prometheus_client.parser import text_string_to_metric_families

BODY = {"model": "m", "max_tokens": 100, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "Explain how umbrellas work."}]}
stop = threading.Event()


def client():
    while not stop.is_set():
        httpx.post("http://127.0.0.1:8081/v1/chat/completions", json=BODY, timeout=60)


def scrape():
    text = httpx.get("http://127.0.0.1:8081/metrics").text
    return {s.name: s.value for f in text_string_to_metric_families(text) for s in f.samples}


threads = [threading.Thread(target=client) for _ in range(8)]
before = scrape()
print("seconds  running  waiting  tokens/s (counter)  busy slots per step")
for t in threads:
    t.start()
last, t0 = before, time.time()
for _ in range(8):
    time.sleep(1)
    m = scrape()
    rate = m["llamacpp:tokens_predicted_total"] - last["llamacpp:tokens_predicted_total"]
    print(f"{time.time() - t0:7.0f}  {m['llamacpp:requests_processing']:7.0f}  "
          f"{m['llamacpp:requests_deferred']:7.0f}  {rate:18.0f}  "
          f"{m['llamacpp:n_busy_slots_per_decode']:19.2f}")
    last = m
stop.set()
for t in threads:
    t.join()
