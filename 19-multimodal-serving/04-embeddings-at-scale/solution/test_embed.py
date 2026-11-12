import embed_llamacpp
import embed_throughput


class Reply:
    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


def test_llamacpp_counts_tokens_and_batches(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append(len(json["input"]))
        return Reply({"usage": {"prompt_tokens": 10 * len(json["input"])}})

    monkeypatch.setattr(embed_llamacpp.httpx, "post", fake_post)
    docs_s, tok_s = embed_llamacpp.run(batch=8, texts_total=32)
    assert calls == [8, 8, 8, 8]                 # 32 documents, 4 requests
    assert docs_s > 0 and tok_s / docs_s == 10   # 10 tokens per document


def test_ollama_version_reports_its_own_token_count(monkeypatch):
    def fake_post(url, json, timeout):
        return Reply({"prompt_eval_count": 7 * len(json["input"])})

    monkeypatch.setattr(embed_throughput.httpx, "post", fake_post)
    _, _, tokens = embed_throughput.run(batch=4, texts_total=16)
    assert tokens == 7 * 16
