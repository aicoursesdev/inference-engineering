"""Time llama-server from process start to its first /health "ok" and its first reply.

    uv run python cold_start.py models/Qwen3-0.6B-Q8_0.gguf models/Qwen3-4B-Q8_0.gguf
Each model is timed twice: the first run may read the file from disk, the second from the OS page cache.
"""
import subprocess
import sys
import time
from pathlib import Path

import httpx

BODY = {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "Hi"}],
        "chat_template_kwargs": {"enable_thinking": False}}


def start_once(path):
    t0 = time.monotonic()
    p = subprocess.Popen(["llama-server", "-m", path, "--port", "8082", "-c", "2048"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while True:
            if p.poll() is not None:
                raise SystemExit(f"llama-server exited early for {path}")
            try:
                if httpx.get("http://127.0.0.1:8082/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        ready = time.monotonic() - t0
        httpx.post("http://127.0.0.1:8082/v1/chat/completions", json=BODY, timeout=60)
        return ready, time.monotonic() - t0
    finally:
        p.terminate()
        p.wait()


for path in sys.argv[1:]:
    gb = Path(path).stat().st_size / 1e9
    for label in ("first", "second"):
        ready, first = start_once(path)
        print(f"{Path(path).name} ({gb:.2f} GB) {label}: healthy after {ready:.1f} s, first reply after {first:.1f} s")
