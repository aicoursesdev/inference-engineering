"""Record one streamed reply from a local llama-server, with timestamps.

    llama-server -m models/Qwen3-0.6B-Q8_0.gguf --port 8081 --jinja   # in another terminal
    uv run python data/record_reply.py

Writes data/recorded/llama-server-reply.json: the raw server-sent events (one per
chunk) with the milliseconds since the request was sent, and how it was made.
This is how the file in the repo was made; running it yourself writes your own.
"""

import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8081"
MODEL_FILE = "Qwen3-0.6B-Q8_0.gguf"
OUT = Path(__file__).resolve().parent / "recorded" / "llama-server-reply.json"
SYSTEM = ("You answer customer emails for Tidewater Goods, a small shop in Kochi that sells rain and outdoor gear. "
          "Returns: within 30 days of delivery, unused, with tags. In India a courier collects the return from the "
          "customer's address for free. Reply in two or three friendly sentences.")
QUESTION = "Hi! The rain jacket I got last week is too big. I haven't worn it and the tags are still on. Can I send it back? I live in Pune."


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def main():
    request = {"model": MODEL_FILE, "temperature": 0, "seed": 0, "max_tokens": 200, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]}
    events = []
    with httpx.Client(timeout=60) as client:
        start = time.perf_counter()
        with client.stream("POST", f"{BASE}/v1/chat/completions", json=request) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data: "):
                    events.append({"t_ms": round((time.perf_counter() - start) * 1000, 1), "data": line[6:]})
    props = httpx.get(f"{BASE}/props").json()
    record = {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "endpoint": f"POST {BASE}/v1/chat/completions (stream)",
        "machine": {"chip": sh("sysctl", "-n", "machdep.cpu.brand_string"), "memory_gb": round(int(sh("sysctl", "-n", "hw.memsize")) / 1024**3),
                    "os": f"{platform.system()} {platform.release()}", "arch": platform.machine()},
        "llama_cpp": {"version": props.get("build_info"), "command": f"llama-server -m models/{MODEL_FILE} --port 8081 --jinja"},
        "model": {"file": MODEL_FILE, "source": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF"},
        "note": "t_ms is milliseconds since the request was sent. Each event's data is the raw server-sent event payload.",
        "request": request,
        "events": events,
    }
    OUT.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(events)} events, first at {events[0]['t_ms']} ms, last at {events[-1]['t_ms']} ms -> {OUT}")


if __name__ == "__main__":
    main()
