"""Time one running engine and add a row to the results ledger.

    uv run python measure_engine.py --engine llama.cpp --version "0.8.0 b11960" \
        --file Qwen3-0.6B-Q8_0.gguf --quant Q8_0 \
        --base http://127.0.0.1:8080/v1 --model qwen3

One engine at a time: start it, run this, stop it. Five timed runs after a
warm-up; the row holds medians, and its note the spread (max minus min),
the engine version and the machine's load.
"""
import argparse
import json
import os
import statistics
import time

import httpx

from servinglab.ledger import record, show

PROMPT = ("Write a long, friendly reply of about 200 words to a customer "
          "in Kochi who asks whether our rain jacket keeps a laptop dry "
          "on a motorbike. Mention the zip, the seams and the hood.")


def one_run(base, model, max_tokens=128):
    body = {"model": model, "stream": True, "temperature": 0,
            "max_tokens": max_tokens,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": PROMPT + " /no_think"}]}
    t0 = time.perf_counter()
    first = last = None
    tokens = 0
    with httpx.stream("POST", base + "/chat/completions", json=body,
                      timeout=120) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            if chunk["choices"] and chunk["choices"][0]["delta"].get("content"):
                last = time.perf_counter()
                first = first or last
    return (first - t0) * 1000, (tokens - 1) / (last - first), tokens


def ledger_row(runs, engine, version, file, quant, machine, load):
    """One ledger row from the timed runs, a list of (ttft_ms, decode_tps,
    tokens): medians for time to first token and decode speed, and a note
    with the spread, the version and the load. Keys are servinglab.ledger's
    columns."""
    raise NotImplementedError(
        "TODO 05.9: build the ledger row from the runs (median time to first "
        "token and decode speed, time per output token as 1000 / decode speed, "
        "and a note with the spread, token count and load), see the "
        "lesson's 'The steps' section")


def main():
    a = argparse.ArgumentParser()
    for name in ["engine", "version", "file", "quant", "base", "model"]:
        a.add_argument("--" + name, required=True)
    a.add_argument("--runs", type=int, default=5)
    a.add_argument("--machine", default="Apple M5 Max laptop, 36 GB")
    a.add_argument("--out", help="a ledger file to use instead of ledger.csv")
    args = a.parse_args()
    load = round(os.getloadavg()[0], 1)  # other jobs share a laptop: record it
    one_run(args.base, args.model)                       # warm-up
    runs = [one_run(args.base, args.model) for _ in range(args.runs)]
    row = ledger_row(runs, args.engine, args.version, args.file, args.quant,
                     args.machine, load)
    record(row, path=args.out)
    show(path=args.out)


if __name__ == "__main__":
    main()
