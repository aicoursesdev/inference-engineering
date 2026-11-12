"""Record the drill's incident 1: a latency regression, from a real llama-server.

    llama-server -m models/Qwen3-0.6B-Q8_0.gguf --port 8082 --jinja -np 4 -c 24576 \
        --metrics --slots     (log to a file)
    uv run python record_latency_incident.py good  8082   # the build before the deploy
    uv run python record_latency_incident.py bad   8082   # the build after it

Four Kodekart-style agents run 8 steps each at once. Every step sends the same long
system prompt plus the growing history. "good" lets the server reuse the cached
start of the prompt (cache_prompt true, llama.cpp's default); "bad" is the deploy
that switched it off in the gateway. Writes one JSON line per request.
"""
import asyncio, json, sys, time
import httpx

MODE, PORT = sys.argv[1], sys.argv[2]
BASE = f"http://127.0.0.1:{PORT}"
OUT = f"incident1-{MODE}.jsonl"
SYSTEM = ("You are Kodebot, a coding agent for Kodekart. Follow the style guide. " +
          " ".join(f"Rule {i}: keep functions short, name things plainly, write a test first." for i in range(1, 121)))
TOOL = " ".join(f"line {i}: def helper_{i}(x): return x + {i}" for i in range(1, 16))
AGENTS, STEPS = 4, 8


async def agent(client, a, rows, t0):
    messages = [{"role": "system", "content": SYSTEM + f" Task #{a}."},
                {"role": "user", "content": f"Task {a}: improve the helpers."}]
    for step in range(1, STEPS + 1):
        body = {"model": "m", "messages": messages, "temperature": 0, "seed": 0, "max_tokens": 24, "stream": True,
                "cache_prompt": MODE == "good", "chat_template_kwargs": {"enable_thinking": False}}
        start, first, text, timings = time.perf_counter(), None, "", {}
        async with client.stream("POST", f"{BASE}/v1/chat/completions", json=body) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                ev = json.loads(line[6:])
                if ev.get("timings"):
                    timings = ev["timings"]
                if ev["choices"] and ev["choices"][0]["delta"].get("content"):
                    first = first or time.perf_counter()
                    text += ev["choices"][0]["delta"]["content"]
        end = time.perf_counter()
        rows.append({"agent": a, "step": step, "sent_s": round(start - t0, 3), "ttft_ms": round((first - start) * 1000, 1) if first else None,
                     "total_ms": round((end - start) * 1000, 1), "prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n"),
                     "prompt_ms": timings.get("prompt_ms"), "predicted_n": timings.get("predicted_n"), "predicted_ms": timings.get("predicted_ms")})
        messages += [{"role": "assistant", "content": text or "ok"}, {"role": "user", "content": f"Tool result: {TOOL}"}]
        await asyncio.sleep(0.5)  # the agent "thinks" half a second


async def main():
    rows = []
    async with httpx.AsyncClient(timeout=300) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(agent(client, a, rows, t0) for a in range(AGENTS)))
        metrics = (await client.get(f"{BASE}/metrics")).text
    rows.sort(key=lambda r: r["sent_s"])
    open(OUT, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    open(f"incident1-{MODE}.metrics.txt", "w").write(metrics)
    t = sorted(r["ttft_ms"] for r in rows)
    print(MODE, "requests", len(rows), "ttft p50", t[len(t) // 2], "p95", t[int(len(t) * .95)], "ms")

asyncio.run(main())
