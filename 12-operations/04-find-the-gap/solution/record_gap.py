"""Record the 12.4 exercise: a request whose first token comes much later than its prefill explains.

    llama-server -m models/Qwen3-0.6B-Q8_0.gguf --port 8084 --jinja -np 1 -c 4096 --metrics   (log to a file)
    uv run python record_gap.py 8084

Three clients send a short chat request each, 50 ms apart, to a server with ONE slot. The server
handles one request at a time, so the second and third wait for a free slot. Writes gap-trace.jsonl:
per request the client's clock (sent_s, ttft_ms, total_ms) and the server's timings.
"""
import asyncio, json, sys, time
import httpx

BASE = f"http://127.0.0.1:{sys.argv[1]}"


async def one(client, i, t0, rows):
    await asyncio.sleep(i * 0.05)
    body = {"model": "m", "temperature": 0, "seed": 0, "max_tokens": 70, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": "You are Tidewater Goods' help desk. Answer in at least five sentences."},
                         {"role": "user", "content": f"Question {i}: how long does a refund take?"}]}
    sent, first, timings = time.perf_counter(), None, {}
    async with client.stream("POST", f"{BASE}/v1/chat/completions", json=body) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                ev = json.loads(line[6:])
                timings = ev.get("timings") or timings
                if first is None and ev["choices"] and ev["choices"][0]["delta"].get("content"):
                    first = time.perf_counter()
    end = time.perf_counter()
    rows.append({"client": i, "sent_s": round(sent - t0, 3), "ttft_ms": round((first - sent) * 1000, 1), "total_ms": round((end - sent) * 1000, 1),
                 "prompt_ms": timings["prompt_ms"], "predicted_n": timings["predicted_n"], "predicted_ms": timings["predicted_ms"]})


async def main():
    rows = []
    async with httpx.AsyncClient(timeout=60) as c:
        await c.post(f"{BASE}/v1/chat/completions", json={"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]})  # warm up
        t0 = time.perf_counter()
        await asyncio.gather(*(one(c, i, t0, rows) for i in range(3)))
    rows.sort(key=lambda r: r["client"])
    open("gap-trace.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(r)

asyncio.run(main())
