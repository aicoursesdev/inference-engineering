"""Module 06: an async load generator for any OpenAI-compatible server, and the report.

    uv run python -m servinglab.bench --url http://localhost:8080 --trace tidewater \\
        --rates 1,2,4,8 --seconds 30 --out results.json
    uv run python -m servinglab.bench --report results.json

It sends requests on a clock (open loop: a slow server does not slow the senders),
streams each reply, and records what a user feels: time to first token (TTFT),
time per output token (TPOT) and total time. A summary gives percentiles and goodput
against a service target (an SLO). Everything an experiment needs to be repeated
(seed, rates, workload, SLO, server, machine) is saved with the numbers.

Prompts are made of plain words, one word of text about one token, so a request of
`input_tokens` is only approximately that long: the server's own count comes back in
each result as `prompt_tokens`.
"""

import argparse
import asyncio
import json
import platform
import random
import time

import httpx
import numpy as np

from .data import traces

# Monsoon's targets (docs/inference-world.md): seconds to first token, seconds per output token.
SLOS = {
    "interactive": {"ttft": 1.0, "tpot": 0.050},
    "agent": {"ttft": 2.0, "tpot": 0.050},
    "long": {"ttft": 8.0, "tpot": 0.080},
}

_WORDS = ("the of and to in is that it for with as was on are by this from at or an be have "
          "not but they which one all their there when who will more if out so up what about "
          "some time can only other new people year work may first also these two then after "
          "use how our well way even want because any day most us good over make water long").split()


def words(n, seed):
    """n plain words, the same every time for the same seed."""
    rng = random.Random(seed)
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def make_requests(rows, max_input=1500, max_output=128, share_prefix=True, seed=0):
    """Turn trace rows into requests. With share_prefix=False no two prompts begin alike,
    which is what a benchmark wants when it must not measure the server's prefix cache."""
    reqs = []
    for i, r in enumerate(rows):
        n_in = min(r["input_tokens"], max_input)
        n_pre = min(r["prefix_tokens"], n_in) if share_prefix else 0
        head = words(n_pre, f"prefix:{r['prefix_id']}") if n_pre else ""
        body = words(n_in - n_pre, f"{seed}:{r['id']}:{i}")
        reqs.append({"id": r["id"], "slo_class": r["slo_class"], "prompt": f"{head} {body}".strip(),
                     "input_tokens": n_in, "max_tokens": min(r["output_tokens"], max_output)})
    return reqs


def arrival_times(rate, n, burstiness=1.0, seed=0):
    """When each of n requests is sent, in seconds from the start. Gaps are gamma-distributed with
    mean 1/rate: burstiness 1 is a Poisson process, below 1 is burstier, above 1 more even
    (the same meaning as vllm bench serve's --burstiness)."""
    rng = np.random.default_rng(seed)
    gaps = rng.gamma(shape=burstiness, scale=1 / (rate * burstiness), size=n)
    return np.cumsum(gaps) - gaps[0]  # the first request goes out at time 0


async def one_request(client, url, model, req, scheduled):
    """Send one streamed request; return its timings (seconds) and token count."""
    out = {"id": req["id"], "slo_class": req["slo_class"], "scheduled": scheduled, "ok": False}
    body = {"model": model, "prompt": req["prompt"], "max_tokens": req["max_tokens"], "stream": True,
            "temperature": 0, "ignore_eos": True, "stream_options": {"include_usage": True}}
    start = time.perf_counter()
    out["sent"], stamps, usage = start, [], None
    try:
        async with client.stream("POST", f"{url}/v1/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                usage = chunk.get("usage") or usage
                if chunk.get("choices") and chunk["choices"][0].get("text"):
                    stamps.append(time.perf_counter())
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        out["error"] = type(e).__name__
        return out
    if not stamps:
        out["error"] = "no tokens"
        return out
    end = time.perf_counter()
    n = (usage or {}).get("completion_tokens") or len(stamps)
    out.update(ok=True, ttft=stamps[0] - start, e2e=end - start, output_tokens=n,
               prompt_tokens=(usage or {}).get("prompt_tokens"),
               tpot=(stamps[-1] - stamps[0]) / (len(stamps) - 1) if len(stamps) > 1 else None)
    return out


async def run(url, model, reqs, rate, burstiness=1.0, seed=0, timeout=300, max_concurrency=None, transport=None):
    """Send reqs at `rate` requests per second (float("inf"): all at once) and wait for every reply.
    `max_concurrency` caps requests in flight, like vllm bench serve's flag: the rest wait in the
    client, so with a cap the rate is only the most the clock will offer. (`transport` lets the
    tests swap the network for a fake server.)"""
    times = arrival_times(rate, len(reqs), burstiness, seed)
    async with httpx.AsyncClient(timeout=timeout, transport=transport,
                                 limits=httpx.Limits(max_connections=None)) as client:
        t0 = time.perf_counter()
        slots = asyncio.Semaphore(max_concurrency or len(reqs))

        async def at(t, r):
            await asyncio.sleep(max(0, t0 + t - time.perf_counter()))
            async with slots:
                res = await one_request(client, url, model, r, t)
            res["lag"] = res["sent"] - t0 - t  # how late the request left: near 0 unless the client or the cap held it back
            return res

        results = await asyncio.gather(*(at(t, r) for t, r in zip(times, reqs)))
        return list(results), time.perf_counter() - t0


def percentile(values, p):
    """The p-th percentile by linear interpolation between neighbours (NumPy's default)."""
    return float(np.percentile(values, p)) if len(values) else None


def meets(res, slo):
    return res["ok"] and res["ttft"] <= slo["ttft"] and (res["tpot"] or 0) <= slo["tpot"]


def summarize(results, seconds, slo, offered_rate=None):
    """Percentiles, throughput and goodput for one run. `seconds` is the run's wall time."""
    ok = [r for r in results if r["ok"]]
    s = {"offered_rate": offered_rate, "requests": len(results), "errors": len(results) - len(ok),
         "seconds": seconds, "slo": slo,
         "request_throughput": len(ok) / seconds,
         "output_tokens_per_s": sum(r["output_tokens"] for r in ok) / seconds,
         "goodput": sum(meets(r, slo) for r in results) / seconds,
         "slo_attainment": sum(meets(r, slo) for r in results) / len(results),
         "max_lag": max((r["lag"] for r in results), default=0)}
    for key in ("ttft", "tpot", "e2e"):
        vals = [r[key] for r in ok if r.get(key) is not None]
        s[key] = {f"p{p}": percentile(vals, p) for p in (50, 95, 99)} | {"mean": float(np.mean(vals)) if vals else None}
    return s


def capacity(summaries, attainment=0.9):
    """The highest offered rate that still met the SLO for `attainment` of requests (DistServe's
    definition of goodput uses 90%), or None if even the lowest failed."""
    good = [s["offered_rate"] for s in summaries if s["slo_attainment"] >= attainment]
    return max(good) if good else None


def sweep(url, model, reqs, rates, slo, burstiness=1.0, seed=0, label="", engine=""):
    """One run per rate, lowest first, each on a fresh copy of the same workload."""
    runs = []
    for rate in rates:
        results, seconds = asyncio.run(run(url, model, reqs, rate, burstiness, seed))
        runs.append({"summary": summarize(results, seconds, slo, rate), "results": results})
    return {"label": label, "engine": engine, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "server": url, "model": model,
            "machine": platform.platform(), "burstiness": burstiness, "seed": seed, "slo": slo,
            "requests_per_run": len(reqs), "runs": runs,
            "capacity_rate": capacity([r["summary"] for r in runs])}


def report(data):
    """A markdown table of the sweep, and the capacity line."""
    slo = data["slo"]
    lines = [f"# {data.get('label') or 'Benchmark'}",
             f"server {data['server']}, model {data['model']}, {data['requests_per_run']} requests per rate, "
             f"burstiness {data['burstiness']}, seed {data['seed']}, {data['created']}",
             f"engine: {data['engine'] or 'not stated'}; machine: {data['machine']}",
             f"SLO: TTFT <= {slo['ttft']} s and TPOT <= {slo['tpot'] * 1000:.0f} ms", "",
             "| rate (req/s) | done/s | tok/s | TTFT p50 | TTFT p95 | TTFT p99 | TPOT p95 (ms) | met SLO | goodput/s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in data["runs"]:
        s = r["summary"]
        f = lambda v, k=1: "-" if v is None else f"{v * k:.2f}"
        lines.append(f"| {s['offered_rate']:g} | {s['request_throughput']:.2f} | {s['output_tokens_per_s']:.0f} | "
                     f"{f(s['ttft']['p50'])} | {f(s['ttft']['p95'])} | {f(s['ttft']['p99'])} | "
                     f"{f(s['tpot']['p95'], 1000)} | {s['slo_attainment']:.0%} | {s['goodput']:.2f} |")
    cap = data["capacity_rate"]
    lines += ["", f"Capacity: {f'{cap:g} requests per second' if cap else 'none: the lowest rate already missed the SLO'}"
              " (the highest rate where 90% of requests met the SLO)."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default="local")
    ap.add_argument("--trace", default="tidewater")
    ap.add_argument("--rates", default="1,2,4,8")
    ap.add_argument("--requests", type=int, default=60, help="requests per rate")
    ap.add_argument("--burstiness", type=float, default=1.0)
    ap.add_argument("--slo", default="interactive", choices=sorted(SLOS))
    ap.add_argument("--max-input", type=int, default=1500)
    ap.add_argument("--max-output", type=int, default=128)
    ap.add_argument("--no-share-prefix", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", default="", help="what is being tested: engine, version and flags, for the report")
    ap.add_argument("--out", help="write the full results here as JSON")
    ap.add_argument("--report", help="print the report of a saved JSON file and stop")
    a = ap.parse_args()
    if a.report:
        print(report(json.load(open(a.report))))
        return
    rows = traces(a.trace)
    random.Random(a.seed).shuffle(rows)
    reqs = make_requests(rows[:a.requests], a.max_input, a.max_output, not a.no_share_prefix, a.seed)
    data = sweep(a.url, a.model, reqs, [float(x) for x in a.rates.split(",")], SLOS[a.slo],
                 a.burstiness, a.seed, label=f"{a.trace} on {a.model}", engine=a.engine)
    print(report(data))
    if a.out:
        json.dump(data, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
