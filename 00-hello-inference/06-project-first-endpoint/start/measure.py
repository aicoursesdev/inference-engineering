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

# TODO 00.6: eight Tidewater help-desk questions (see the lesson's 'Build the tool' section)
QUESTIONS = []
# TODO 00.6: a system line asking for two or three friendly sentences
SYSTEM = ""


def body(question, model):
    raise NotImplementedError(
        "TODO 00.6: return the request body for one question: temperature 0, "
        "max_tokens 120, thinking switched off, the system line and the "
        "question, see the lesson's 'Build the tool' section")


def load(url, bodies, in_flight, timed=timed_stream):
    """Send every body, never more than in_flight at once; one timing each."""
    raise NotImplementedError(
        "TODO 00.6: call timed(url, bodies[0]) once and throw it away, then "
        "send every body with a ThreadPoolExecutor of in_flight workers and "
        "return the timings in order, see the lesson's 'Build the tool' section")


def ledger_row(reqs, model, hardware, change, in_flight):
    """One ledger row, scored against Monsoon's `interactive` targets."""
    raise NotImplementedError(
        "TODO 00.6: score reqs with summary(reqs, SLO['interactive']) and "
        "return a dict with the ledger's lesson, engine, model, hardware, "
        "change, requests, ttft_ms, tpot_ms, tok_per_s, goodput_rps and note "
        "(p50 times in milliseconds), see the lesson's 'Build the tool' section")


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
