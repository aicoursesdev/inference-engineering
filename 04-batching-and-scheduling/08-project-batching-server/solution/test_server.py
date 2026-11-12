"""The server, offline: the engine loop and the HTTP handlers on the tiny random model."""

import asyncio
import json
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from servinglab.engine import Config, Model, generate
from servinglab.engine.weights import read_safetensors
from servinglab.kv import PagedCache
from servinglab.scheduler import Scheduler, Request

from server import EngineLoop, build

TINY = next(p for p in Path(__file__).parents if (p / "servinglab").is_dir()) / "servinglab/tests/data/tiny-llama"


class TinyTok:
    def encode(self, text, add_special_tokens=False):
        return type("Encoding", (), {"ids": [3 + ord(c) % 90 for c in text]})()

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{i}>" for i in ids)


def tiny_model():
    config = Config.from_json(json.loads((TINY / "config.json").read_text()))
    config = Config(**{**config.__dict__, "eos": (-1,)})
    return Model(config, read_safetensors(TINY / "model.safetensors"), TinyTok(), TINY)


@pytest.fixture(autouse=True)
def step_is_written():
    """Fail fast, with the TODO, instead of waiting on a thread that died."""
    model = tiny_model()
    cache = PagedCache(model.config, 8, 4)
    EngineLoop(model, Scheduler(alloc=cache.alloc), cache).step()    # idle: returns at once


def test_engine_loop_streams_every_token_and_a_full_queue_is_refused():
    config = Config.from_json(json.loads((TINY / "config.json").read_text()))
    config = Config(**{**config.__dict__, "eos": (-1,)})
    model = Model(config, read_safetensors(TINY / "model.safetensors"), TinyTok(), TINY)
    cache = PagedCache(config, 64, 4)
    sched = Scheduler(alloc=cache.alloc, token_budget=8, max_seqs=2, max_queue=1)
    engine = EngineLoop(model, sched, cache)
    engine.start()

    async def go():
        loop = asyncio.get_running_loop()
        outs = []
        for i in range(4):
            q = asyncio.Queue()
            r = Request(f"r{i}", [5 + i, 9, 11, 3], 6)
            engine.submit(r, loop, q)
            outs.append(q)
        results = []
        for q in outs:
            items = []
            while True:
                kind, _ = item = await asyncio.wait_for(q.get(), 30)
                items.append(kind)
                if kind in ("done", "rejected"):
                    break
            results.append(items)
        return results

    results = asyncio.run(go())
    assert any(r == ["rejected"] for r in results)
    assert any(r == ["token"] * 6 + ["done"] for r in results)


def test_a_completion_returns_the_plain_engines_tokens_and_the_queue_drains():
    model = tiny_model()
    ids = model.encode("monsoon")
    plain = model.decode(list(generate(model, ids, 6, temperature=0)))
    with TestClient(build(model=model, blocks=64, budget=8, prefix_cache=False)) as client:
        reply = client.post("/v1/completions", json={"prompt": "monsoon", "max_tokens": 6,
                                                      "temperature": 0}).json()
        assert reply["choices"][0]["text"] == plain
        assert reply["usage"] == {"prompt_tokens": len(ids), "completion_tokens": 6}
        stats = client.get("/stats").json()
        assert stats["running"] == 0 and stats["waiting"] == 0 and stats["steps"] > 0


def test_a_streamed_completion_sends_events_then_done():
    model = tiny_model()
    with TestClient(build(model=model, blocks=64, budget=8, prefix_cache=False)) as client:
        text = client.post("/v1/completions", json={"prompt": "kerala", "max_tokens": 4,
                                                     "stream": True}).text
    events = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert len([e for e in events[:-1] if json.loads(e)["choices"]]) == 4   # the usage event has none
