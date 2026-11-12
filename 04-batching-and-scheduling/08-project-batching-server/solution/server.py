"""A continuous-batching mini server: your engine, allocator and scheduler behind HTTP.

    uv run --extra service python server.py --mode continuous --port 8000

Speaks the start of the OpenAI completions API, enough for servinglab.bench:

    POST /v1/completions  {"prompt": "...", "max_tokens": 32, "stream": true,
                           "temperature": 0, "ignore_eos": false}
    GET  /stats           what the scheduler is doing right now

One thread runs the engine loop (schedule, run the step, update); the web
handlers only hand requests over and stream the tokens that come back.
Answers 429 when the queue is full (admission control).
"""

import argparse
import asyncio
import itertools
import json
import queue
import threading
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from servinglab.engine import load
from servinglab.kv import PagedCache
from servinglab.scheduler import Executor, Request, Scheduler


class EngineLoop(threading.Thread):
    def __init__(self, model, sched, cache):
        super().__init__(daemon=True)
        self.model, self.sched, self.executor = model, sched, Executor(model, cache)
        self.inbox, self.outbox = queue.Queue(), {}
        self.wake = threading.Event()
        self.steps = 0

    def submit(self, req, loop, out):
        self.outbox[req.id] = (loop, out, 0)
        self.inbox.put(req)
        self.wake.set()

    def run(self):
        while True:
            self.step()

    def step(self):
        """One turn of the loop: take new requests, then run one step."""
        while True:                                          # take new requests
            try:
                req = self.inbox.get_nowait()
            except queue.Empty:
                break
            if not self.sched.submit(req):
                self._send(req.id, ("rejected", None))
        if self.sched.idle:
            self.wake.wait(0.05)
            self.wake.clear()
            return
        plan = self.sched.schedule()
        if not plan:                                          # nothing fits right now
            time.sleep(0.001)
            return
        sampled = self.executor.run(plan)
        self.sched.update(plan, time.perf_counter(), sampled)
        self.steps += 1
        for req, _ in plan:
            if req.id in sampled:
                tok = sampled[req.id]
                self._send(req.id, ("token", self.model.decode([tok]) if tok is not None else ""))
            if req.status == "done":
                self._send(req.id, ("done", None))

    def _send(self, rid, item):
        loop, out, _ = self.outbox[rid]
        loop.call_soon_threadsafe(out.put_nowait, item)
        if item[0] in ("done", "rejected"):
            del self.outbox[rid]


def build(mode="continuous", budget=64, blocks=512, max_seqs=8, max_queue=None, prefix_cache=True,
          model=None):
    model = model or load("SmolLM2-135M-Instruct")      # tests pass a tiny model
    cache = PagedCache(model.config, blocks, 16)
    sched = Scheduler(alloc=cache.alloc, mode=mode, token_budget=budget, chunked=True,
                      max_seqs=max_seqs, max_queue=max_queue, prefix_cache=prefix_cache)
    engine = EngineLoop(model, sched, cache)
    engine.start()
    app, ids = FastAPI(), itertools.count()

    @app.get("/stats")
    def stats():
        return {"running": len(sched.running), "waiting": len(sched.waiting), "steps": engine.steps,
                "preemptions": sched.preemptions, "cache_hit_rate": round(cache.alloc.hit_rate, 3),
                "free_blocks": cache.alloc.free_blocks}

    @app.post("/v1/completions")
    async def completions(body: dict):
        prompt_ids = model.encode(body["prompt"])
        req = Request(f"req{next(ids)}", prompt_ids, int(body.get("max_tokens", 16)))
        req.ignore_eos = bool(body.get("ignore_eos", False))
        out = asyncio.Queue()
        engine.submit(req, asyncio.get_running_loop(), out)
        first = await out.get()
        if first[0] == "rejected":
            return JSONResponse({"error": "server busy, try later"}, status_code=429)

        async def pieces():
            item = first
            while item[0] != "done":
                yield item[1]
                item = await out.get()

        if body.get("stream"):
            async def sse():
                async for text in pieces():
                    yield "data: " + json.dumps({"choices": [{"text": text}]}) + "\n\n"
                usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": req.generated}
                yield "data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")
        text = "".join([t async for t in pieces()])
        return {"choices": [{"text": text}],
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": req.generated}}

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="continuous", choices=["static", "continuous"])
    ap.add_argument("--budget", type=int, default=64, help="tokens per step")
    ap.add_argument("--blocks", type=int, default=512)
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--max-queue", type=int, default=None)
    ap.add_argument("--no-prefix-cache", action="store_true", help="benchmarks that repeat prompts need this")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(build(a.mode, a.budget, a.blocks, a.max_seqs, a.max_queue, not a.no_prefix_cache), port=a.port, log_level="warning")
