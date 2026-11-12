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
        raise NotImplementedError(
            "TODO 04.8: move requests from self.inbox into "
            "sched.submit (send ('rejected', None) when refused); "
            "when the scheduler is idle wait on self.wake and return;"
            " else plan = schedule(), self.executor.run(plan), "
            "sched.update(plan, time.perf_counter(), sampled), count "
            "the step and send each sampled token ('token', text) and"
            " ('done', None) when a request finishes, see the "
            "lesson's 'The design' section")

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
        raise NotImplementedError(
            "TODO 04.8: tokenise the prompt, make a Request, call "
            "engine.submit, answer 429 when the first item is "
            "'rejected', else stream server-sent events or return the"
            " whole text, with usage, see the lesson's 'The design' "
            "section")

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
