"""A streaming gateway in front of llama-server (lesson 10.3).

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
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8081")
MAX_IN_FLIGHT = int(os.environ.get("MAX_IN_FLIGHT", "2"))   # as many as the engine has slots
MAX_WAITING = int(os.environ.get("MAX_WAITING", "4"))       # beyond this, say 503
FIRST_BYTE_S = float(os.environ.get("FIRST_BYTE_S", "30"))  # wait for the first chunk
GAP_S = float(os.environ.get("GAP_S", "10"))                # longest silence between chunks
DEADLINE_S = float(os.environ.get("DEADLINE_S", "120"))     # whole request

app = FastAPI()
client = httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(5.0, read=None))
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
    body = await request.json()
    if waiting >= MAX_WAITING:  # backpressure: a short, honest refusal
        raise NotImplementedError("TODO 10.3: refuse at once with a 503 and a Retry-After header, see the lesson's 'Backpressure: say no early' section")
    waiting += 1
    try:
        await asyncio.wait_for(slots.acquire(), timeout=FIRST_BYTE_S)
    except TimeoutError:
        return JSONResponse({"error": "queue timeout"}, 504)
    finally:
        waiting -= 1
    try:
        req = client.build_request("POST", request.url.path, json=body)
        upstream = await asyncio.wait_for(client.send(req, stream=True), FIRST_BYTE_S)
    except (httpx.HTTPError, TimeoutError):
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
            # TODO 10.3: close the upstream connection and release the slot, whatever ended the stream
            raise NotImplementedError("TODO 10.3: close the upstream connection and release the slot, see the lesson's 'Hanging up must stop the work' section")

    return StreamingResponse(relay(), media_type="text/event-stream")
