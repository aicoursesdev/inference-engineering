"""Ask the local llama-server a question and print each token as it arrives.

    llama-server -hf Qwen/Qwen3-0.6B-GGUF:Q8_0 --port 8081
    uv run python stream_demo.py
"""

import json
import os
import sys
import time

import httpx

PORT = os.environ.get("PORT", "8081")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
BODY = {
    "model": "Qwen3-0.6B-Q8_0",
    "messages": [
        {"role": "system", "content": "You answer customer emails for "
         "Tidewater Goods, a rain-gear shop in Kochi. Reply in two or "
         "three friendly sentences."},
        {"role": "user", "content": "Hi! The rain jacket I got last week "
         "is too big. Can I send it back? I live in Pune."},
    ],
    "temperature": 0,
    "max_tokens": 100,
    "stream": True,
    "chat_template_kwargs": {"enable_thinking": False},
}

sent = time.perf_counter()
with httpx.stream("POST", URL, json=BODY, timeout=60) as reply:
    for line in reply.iter_lines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if not chunk.get("choices"):  # an error chunk has no choices
            continue
        delta = chunk["choices"][0]["delta"]
        if delta.get("content"):
            ms = (time.perf_counter() - sent) * 1000
            print(f"{ms:7.1f} ms  {delta['content']!r}")
            sys.stdout.flush()
