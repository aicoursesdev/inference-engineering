"""Time one streamed chat reply from any OpenAI-compatible server.

Works the same against trtllm-serve, vLLM, SGLang or llama-server:
    python ask.py http://localhost:8000 <model name>
"""
import json, sys, time, urllib.request

base, model = sys.argv[1].rstrip("/"), sys.argv[2]
body = json.dumps({"model": model, "stream": True, "max_tokens": 64,
                   "temperature": 0,
                   "messages": [{"role": "user",
                                 "content": "Name three monsoon rivers."}]})
req = urllib.request.Request(base + "/v1/chat/completions", body.encode(),
                             {"Content-Type": "application/json"})
t0 = time.perf_counter()
first, chunks, text = None, 0, ""
with urllib.request.urlopen(req, timeout=120) as resp:
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: ") or line.endswith("[DONE]"):
            continue
        delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
        if delta:
            first = first or time.perf_counter() - t0
            chunks += 1
            text += delta
total = time.perf_counter() - t0
print(text.strip()[:200])
print(f"TTFT {first*1000:.0f} ms, {chunks} chunks in {total:.2f} s")
