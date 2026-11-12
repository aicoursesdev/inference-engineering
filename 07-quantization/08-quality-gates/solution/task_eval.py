"""A 40-question task eval against a running llama-server.

    uv run python task_eval.py http://127.0.0.1:8080/v1 > answers.json

An answer counts when the gold string appears in the reply, ignoring case.
Temperature 0, thinking off, 64 tokens, "answer only": cheap, repeatable, and crude.
"""
import json
import sys
from pathlib import Path

import httpx

from servinglab.quality import accuracy


def ask(base, question):
    r = httpx.post(base + "/chat/completions", timeout=60, json={
        "model": "m", "temperature": 0, "max_tokens": 64,
        "messages": [{"role": "user",
                      "content": question + " Reply with only the answer. /no_think"}]})
    return r.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    items = json.loads((Path(__file__).parent / "tasks.json").read_text())
    replies = [ask(sys.argv[1], it["q"]) for it in items]
    hits = [it["gold"].lower() in rep.lower() for it, rep in zip(items, replies)]
    print(json.dumps({"accuracy": accuracy(hits, [True] * len(hits)),
                      "replies": replies}))
