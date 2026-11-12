import json

import measure_stream


class FakeStream:
    """Stands in for httpx.stream: three content chunks, then the usage chunk."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        for text in ("a", "b", "c"):
            yield "data: " + json.dumps({"choices": [{"delta": {"content": text}}]})
        yield "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 3}})
        yield "data: [DONE]"


def test_one_run_reads_first_token_speed_and_the_servers_token_count(monkeypatch):
    monkeypatch.setattr(measure_stream.httpx, "stream", lambda *a, **k: FakeStream())
    ttft, tps, tokens = measure_stream.one_run("http://fake", "m")
    assert tokens == 3 and ttft >= 0 and tps > 0
