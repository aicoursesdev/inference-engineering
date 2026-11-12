"""The gateway with auth, token-budget rate limits and fallback (lesson 10.4).

Builds on 10.3's gateway.py. KEYS="sk-tidewater:tidewater:6000,sk-kodekart:kodekart:60000" sets who may call.

    llama-server -m models/Qwen3-0.6B-Q8_0.gguf --port 8081 -np 2 --metrics
    uv sync --extra service
    UPSTREAM=http://127.0.0.1:8081 uv run uvicorn gateway:app --port 8090

It forwards /v1/chat/completions, streams the reply, and does four jobs a bare
engine does not: stop the upstream work when the client leaves, give up on a
silent upstream, cap how many requests are in flight, and refuse when the line is too long.
"""

import asyncio
import json
import os
import time

import httpx
from fastapi import FastAPI, Request
from policy import Keys, digest
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAMS = os.environ.get("UPSTREAMS", "http://127.0.0.1:8081").split(",")  # first is preferred
MAX_IN_FLIGHT = int(os.environ.get("MAX_IN_FLIGHT", "2"))   # as many as the engine has slots
MAX_WAITING = int(os.environ.get("MAX_WAITING", "4"))       # beyond this, say 503
FIRST_BYTE_S = float(os.environ.get("FIRST_BYTE_S", "30"))  # wait for the first chunk
GAP_S = float(os.environ.get("GAP_S", "10"))                # longest silence between chunks
DEADLINE_S = float(os.environ.get("DEADLINE_S", "120"))     # whole request

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=None))
keys = Keys({digest(k): {"customer": c, "tokens_per_min": int(n)}
             for k, c, n in (e.split(":") for e in os.environ.get("KEYS", "").split(",") if e)})
slots = asyncio.Semaphore(MAX_IN_FLIGHT)
waiting = 0
cancelled = 0  # for the demo: how many clients left mid-reply


@app.get("/stats")
def stats():
    return {"waiting": waiting, "cancelled": cancelled}


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
async def chat(request: Request):
    global waiting
    # AUTH=off is for a load test behind a private tunnel only: bench.py sends no key
    who = ({"customer": "loadtest", "tokens_per_min": 10**9} if os.environ.get("AUTH") == "off"
           else keys.check(request.headers.get("authorization")))
    if who is None:
        return JSONResponse({"error": "invalid api key"}, 401)
    body = await request.json()
    body["max_tokens"] = min(body.get("max_tokens", 256), 512)  # safe default: never unbounded
    wait = keys.charge(who, body["max_tokens"])  # charge the most the request may use
    if wait:
        return JSONResponse({"error": "rate limited"}, 429, headers={"Retry-After": str(int(wait) + 1)})
    if waiting >= MAX_WAITING:  # backpressure: a short, honest refusal
        return JSONResponse({"error": "busy"}, 503, headers={"Retry-After": "2"})
    waiting += 1
    try:
        await asyncio.wait_for(slots.acquire(), timeout=FIRST_BYTE_S)
    except TimeoutError:
        return JSONResponse({"error": "queue timeout"}, 504)
    finally:
        waiting -= 1
    upstream = None
    for base in UPSTREAMS:  # fallback: next upstream, but only before any byte has been sent
        try:
            req = client.build_request("POST", base + request.url.path, json=body)
            upstream = await asyncio.wait_for(client.send(req, stream=True), FIRST_BYTE_S)
            if upstream.status_code < 500:
                break
            await upstream.aclose()
            upstream = None
        except (httpx.HTTPError, TimeoutError):
            upstream = None
    if upstream is None:
        slots.release()
        return JSONResponse({"error": "upstream unavailable"}, 502)
    if not body.get("stream"):  # not streaming: read it all, still bounded
        try:
            data = await asyncio.wait_for(upstream.aread(), DEADLINE_S)
            return JSONResponse(content=json.loads(data), status_code=upstream.status_code)
        finally:
            await upstream.aclose()
            slots.release()

    async def relay():
        global cancelled
        start = time.monotonic()
        chunks = upstream.aiter_raw()
        try:
            while True:
                left = DEADLINE_S - (time.monotonic() - start)
                try:
                    chunk = await asyncio.wait_for(anext(chunks), min(GAP_S, left))
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    yield b'data: {"error": "timeout"}\n\n'
                    return
                yield chunk
        except asyncio.CancelledError:  # the client hung up
            cancelled += 1
            raise
        finally:
            await upstream.aclose()  # closing the socket is what tells llama-server to stop
            slots.release()

    return StreamingResponse(relay(), media_type="text/event-stream")
