"""Offline tests for lesson 10.4: the gateway's 401 and 429, with a fake engine behind it."""
import asyncio

import httpx
from fastapi.testclient import TestClient

import gateway
from policy import Keys, digest


def setup(monkeypatch):
    handler = lambda request: httpx.Response(200, json={"choices": []})
    monkeypatch.setattr(gateway, "client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(gateway, "slots", asyncio.Semaphore(gateway.MAX_IN_FLIGHT))
    monkeypatch.setattr(gateway, "keys", Keys({digest("sk-tw"): {"customer": "tidewater", "tokens_per_min": 600}}))
    return TestClient(gateway.app)


def call(c, key=None):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return c.post("/v1/chat/completions", json={"max_tokens": 200}, headers=headers)


def test_no_key_and_a_wrong_key_get_401(monkeypatch):
    c = setup(monkeypatch)
    assert call(c).status_code == 401
    assert call(c, "nope").status_code == 401


def test_three_calls_pass_and_the_fourth_is_429_with_retry_after(monkeypatch):
    c = setup(monkeypatch)
    assert [call(c, "sk-tw").status_code for _ in range(3)] == [200, 200, 200]
    r = call(c, "sk-tw")
    assert r.status_code == 429 and 20 <= int(r.headers["retry-after"]) <= 21
