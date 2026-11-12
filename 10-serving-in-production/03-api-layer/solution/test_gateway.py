"""Offline tests for lesson 10.3: a fake llama-server (httpx.MockTransport) stands in for the real one."""
import asyncio
import json

import httpx
from fastapi.testclient import TestClient

import gateway


class Chunks(httpx.AsyncByteStream):
    """A streamed reply; `hang` makes it go silent after the last chunk, like a long generation."""

    def __init__(self, chunks, hang=False):
        self.chunks, self.hang, self.closed = chunks, hang, False

    async def __aiter__(self):
        for c in self.chunks:
            yield c
        if self.hang:
            await asyncio.sleep(3600)

    async def aclose(self):
        self.closed = True


def fake_upstream(monkeypatch, stream):
    handler = lambda request: httpx.Response(200, stream=stream)
    monkeypatch.setattr(gateway, "client", httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://fake"))
    monkeypatch.setattr(gateway, "slots", asyncio.Semaphore(gateway.MAX_IN_FLIGHT))
    monkeypatch.setattr(gateway, "waiting", 0)
    monkeypatch.setattr(gateway, "cancelled", 0)


def test_a_streamed_reply_passes_through_and_frees_the_slot(monkeypatch):
    stream = Chunks([b"data: one\n\n", b"data: two\n\n"])
    fake_upstream(monkeypatch, stream)
    r = TestClient(gateway.app).post("/v1/chat/completions", json={"stream": True})
    assert r.content == b"data: one\n\ndata: two\n\n"
    assert stream.closed and gateway.slots._value == gateway.MAX_IN_FLIGHT


def test_a_full_line_is_refused_at_once_with_retry_after(monkeypatch):
    fake_upstream(monkeypatch, Chunks([b"x"]))
    monkeypatch.setattr(gateway, "waiting", gateway.MAX_WAITING)
    r = TestClient(gateway.app).post("/v1/chat/completions", json={})
    assert r.status_code == 503 and "retry-after" in r.headers


def test_hanging_up_closes_the_upstream_and_frees_the_slot(monkeypatch):
    stream = Chunks([b"data: one\n\n"], hang=True)
    fake_upstream(monkeypatch, stream)
    body, sent, first = json.dumps({"stream": True}).encode(), [], asyncio.Event()

    async def receive():
        if not sent:
            sent.append(1)
            return {"type": "http.request", "body": body, "more_body": False}
        await first.wait()                      # the client leaves after the first chunk
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first.set()

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "query_string": b"",
             "headers": [(b"content-type", b"application/json")], "http_version": "1.1",
             "scheme": "http", "server": ("test", 80), "client": ("test", 1), "root_path": ""}
    asyncio.run(asyncio.wait_for(gateway.app(scope, receive, send), 5))
    assert stream.closed, "the connection to the engine was not closed"
    assert gateway.slots._value == gateway.MAX_IN_FLIGHT
    assert gateway.cancelled == 1
