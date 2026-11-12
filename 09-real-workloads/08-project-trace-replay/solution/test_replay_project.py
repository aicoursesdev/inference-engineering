"""Offline checks for the project: prompts extend each other, the ceiling is
what the trace says, and the cost drops when the cache hits."""

import asyncio
import json

import httpx

from replay import (Prices, Row, build_requests, cached_of, replay, report,
                    trace_hit_rate)


def test_each_kodekart_step_extends_the_last():
    steps = [r for r in build_requests("kodekart", scale=0.05, tasks=1)]
    assert len(steps) > 4
    for a, b in zip(steps, steps[1:]):
        assert b.prompt.startswith(a.prompt)


def test_the_ceiling_for_kodekart_is_about_92_percent():
    assert abs(trace_hit_rate(build_requests("kodekart")) - 0.922) < 0.001


def test_cached_tokens_cost_less():
    p = Prices(input=1.0, cached_input=0.25, output=2.0)
    cold = [Row("a", "s", prompt_tokens=1000, output_tokens=10)]
    warm = [Row("a", "s", prompt_tokens=1000, cached_tokens=900, output_tokens=10)]
    assert report(warm, p)["cost_per_task_usd"] < report(cold, p)["cost_per_task_usd"]


def test_cached_tokens_come_from_either_server_dialect():
    assert cached_of({"usage": {"prompt_tokens_details": {"cached_tokens": 7}}}) == 7
    assert cached_of({"usage": {}, "timings": {"cache_n": 5}}) == 5
    assert cached_of({}) == 0


def sse(prompt, cached, text="{}"):
    chunks = [{"choices": [{"delta": {"content": text}}]},
              {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": 2,
                                        "prompt_tokens_details": {"cached_tokens": cached}}}]
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def test_replay_runs_a_sessions_steps_in_order_and_records_what_the_server_says():
    order = []

    def handler(request):
        order.append(json.loads(request.content)["messages"][0]["content"])
        return httpx.Response(200, content=sse(100, 40).encode())
    reqs = build_requests("kodekart", scale=0.02, tasks=2)
    rows = asyncio.run(replay(reqs, "http://x", "m", concurrency=2,
                              transport=httpx.MockTransport(handler)))
    assert [r.id for r in rows] == [r.id for r in reqs]
    assert all(r.cached_tokens == 40 and r.ttft_s > 0 and not r.error for r in rows)
    first = [r.prompt for r in reqs if r.session == reqs[0].session]
    assert [o for o in order if o in first] == first


def test_a_failing_server_becomes_error_rows_not_a_crash():
    reqs = build_requests("kodekart", scale=0.02, tasks=1)
    rows = asyncio.run(replay(reqs, "http://x", "m", transport=httpx.MockTransport(
        lambda request: httpx.Response(500))))
    assert all(r.error for r in rows)
    assert report(rows, Prices(input=1, output=1))["errors"] == len(reqs)


def test_report_prices_uncached_cached_and_output_tokens_apart():
    rows = [Row("a", "s", prompt_tokens=1000, cached_tokens=600, output_tokens=100, ttft_s=0.2)]
    rep = report(rows, Prices(input=1.0, cached_input=0.1, output=2.0))
    assert rep["hit_rate"] == 0.6 and rep["ttft_p50_s"] == 0.2
    assert abs(rep["cost_usd"] - (400 * 1.0 + 600 * 0.1 + 100 * 2.0) / 1e6) < 1e-12
