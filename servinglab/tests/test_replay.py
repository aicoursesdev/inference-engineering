"""Trace replay (Module 09). Offline: a pretend server, no network."""

import asyncio
import json

import httpx

from servinglab.replay import (Prices, build_requests, filler, replay, report,
                               trace_hit_rate)


def test_filler_is_deterministic_and_the_right_length():
    assert filler("a", 50) == filler("a", 50)
    assert len(filler("a", 50).split()) == 50 and filler("a", 5) != filler("b", 5)


def test_kodekart_steps_extend_the_previous_prompt():
    reqs = build_requests("kodekart", scale=0.05, tasks=3)
    by = {}
    for r in reqs:
        by.setdefault(r.session, []).append(r)
    assert len(by) == 3
    for steps in by.values():
        for a, b in zip(steps, steps[1:]):
            assert b.prompt.startswith(a.prompt)


def test_the_trace_allows_92_percent_reuse_for_kodekart():
    reqs = build_requests("kodekart", scale=1.0)
    assert len(reqs) == 641 and abs(trace_hit_rate(reqs) - 0.9224) < 0.001


def fake_server(reuse):
    """Echoes usage; `reuse` says how much of a repeated prefix it keeps."""
    seen = []

    def handler(request):
        body = json.loads(request.content)
        text = body["messages"][0]["content"]
        n = len(text.split())
        hit = max((len(p.split()) for p in seen if text.startswith(p)),
                  default=0) if reuse and body.get("cache_prompt", True) else 0
        seen.append(text)
        sse = (f'data: {json.dumps({"choices": [{"delta": {"content": "{}"}}]})}\n\n'
               f'data: {json.dumps({"choices": [], "usage": {"prompt_tokens": n, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": hit}}})}\n\n'
               "data: [DONE]\n\n")
        return httpx.Response(200, content=sse.encode())
    return httpx.MockTransport(handler)


def test_replay_reports_hit_rate_and_cost():
    reqs = build_requests("kodekart", scale=0.05, tasks=2)
    prices = Prices(input=1.0, cached_input=0.1, output=2.0)
    run = lambda reuse, cache=True: report(asyncio.run(replay(
        reqs, "http://x", "m", concurrency=2, cache=cache,
        transport=fake_server(reuse), schema={"type": "object"})), prices)
    on, off = run(True), run(True, cache=False)
    assert on["hit_rate"] > 0.5 and off["hit_rate"] == 0
    assert on["cost_per_task_usd"] < off["cost_per_task_usd"]
    assert on["requests"] == len(reqs) and on["errors"] == 0


def test_a_list_of_models_is_dealt_out_one_per_session():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, content=b"data: [DONE]\n\n")
    reqs = build_requests("kodekart", scale=0.02, tasks=3)
    asyncio.run(replay(reqs, "http://x", ["a", "b"], concurrency=3,
                       transport=httpx.MockTransport(handler)))
    assert set(seen) == {"a", "b"} and len(seen) == len(reqs)
