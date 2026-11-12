"""Measure a local OpenAI-style endpoint and add a row to the ledger.

    llama-server -hf Qwen/Qwen3-0.6B-GGUF:Q8_0 --port 8081
    uv run python measure.py --change "baseline, Q8_0, 4 in flight"

Sends Tidewater-style help-desk questions, a few at a time, times every
token, scores the run against Monsoon's `interactive` targets and records
one ledger row. Also saves the raw timings to run.json.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from servinglab.ledger import record, show
from servinglab.metrics import SLO, summary, timed_stream

QUESTIONS = [
    "Can I return a rain jacket I bought last week? It is too big.",
    "How long does delivery to Pune take?",
    "Is the trekking poncho waterproof or only water resistant?",
    "My order says delivered but I have not got it. What now?",
    "Do you have gumboots in size 9?",
    "How do I wash a rain jacket without ruining the coating?",
    "Can I change the delivery address after ordering?",
    "Do you ship to Goa? How much does it cost?",
]
SYSTEM = ("You answer customer emails for Tidewater Goods, a rain-gear shop "
          "in Kochi. Reply in two or three friendly sentences.")


def body(question, model):
    return {"model": model, "temperature": 0, "max_tokens": 120,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": question}]}


def load(url, bodies, in_flight, timed=timed_stream):
    """Send every body, never more than in_flight at once; one timing each."""
    timed(url, bodies[0])  # warm up: its timing is thrown away
    with ThreadPoolExecutor(in_flight) as pool:
        return list(pool.map(lambda b: timed(url, b), bodies))


def ledger_row(reqs, model, hardware, change, in_flight):
    """One ledger row, scored against Monsoon's `interactive` targets."""
    s = summary(reqs, SLO["interactive"])
    return {"lesson": "00.6", "engine": "llama.cpp", "model": model,
            "hardware": hardware, "change": change,
            "requests": s["requests"],
            "ttft_ms": round(s["ttft_p50"] * 1000, 1),
            "tpot_ms": round(s["tpot_p50"] * 1000, 1),
            "tok_per_s": round(s["tokens_per_s"]),
            "goodput_rps": round(s["goodput_rps"], 2),
            "note": f"{in_flight} in flight; "
                    f"{s['good_fraction']:.0%} met interactive"}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--port", default="8081")
    a.add_argument("--model", default="Qwen3-0.6B-Q8_0")
    a.add_argument("--requests", type=int, default=24)
    a.add_argument("--in-flight", type=int, default=4)
    a.add_argument("--change", default="baseline")
    a.add_argument("--hardware", default="M5-Max")
    args = a.parse_args()

    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    bodies = [body(QUESTIONS[i % len(QUESTIONS)], args.model)
              for i in range(args.requests)]
    reqs = load(url, bodies, args.in_flight)

    json.dump([{"sent": r.sent, "token_times": r.token_times} for r in reqs],
              open("run.json", "w"))
    record(ledger_row(reqs, args.model, args.hardware, args.change,
                      args.in_flight))
    show()


if __name__ == "__main__":
    main()
