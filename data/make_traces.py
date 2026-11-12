"""Make the course's request traces. SYNTHETIC: invented customers, invented numbers.

    uv run python data/make_traces.py

Writes data/traces/{tidewater,kodekart,nyaya,batch,mixed}.jsonl. The seed is fixed, so
the same command always writes the same bytes. Distributions are in data/README.md.
Times are seconds since midnight IST on one day; one request per line.
"""

import json
from pathlib import Path

import numpy as np

SEED = 20261112
OUT = Path(__file__).resolve().parent / "traces"
DAY = 86400


def lognormal(rng, median, sigma, lo, hi, n=None):
    """Integer tokens from a log-normal with this median, clipped to [lo, hi]."""
    x = rng.lognormal(np.log(median), sigma, n)
    return np.clip(np.rint(x), lo, hi).astype(int)


def tidewater_rate(sec):
    """Requests per second at this time of day (IST): quiet nights, busy 9:00-18:00,
    a lunch dip, and a festival-sale spike (4x) from 12:00 to 12:30."""
    h = sec / 3600
    if 12 <= h < 12.5:
        return 0.1
    if 13 <= h < 14:
        return 0.018
    if 9 <= h < 18:
        return 0.025
    if 7 <= h < 9 or 18 <= h < 22:
        return 0.008
    return 0.002


def tidewater(rng):
    times = []
    for minute in range(DAY // 60):  # Poisson count per minute, uniform inside it
        n = rng.poisson(tidewater_rate(minute * 60 + 30) * 60)
        times += list(minute * 60 + rng.uniform(0, 60, n))
    times.sort()
    n = len(times)
    inp = lognormal(rng, 350, 0.5, 240, 1500, n)  # 220 of these are the shared system prompt
    out = lognormal(rng, 120, 0.6, 10, 600, n)
    return [dict(arrival_s=t, customer="tidewater", slo_class="interactive", input_tokens=int(i),
                 output_tokens=int(o), prefix_id="tw-system", prefix_tokens=220)
            for t, i, o in zip(times, inp, out)]


def kodekart(rng, tasks=40):
    rows = []
    starts = np.sort(rng.uniform(8 * 3600, 20 * 3600, tasks))
    for task, t in enumerate(starts):
        steps = int(rng.integers(5, 26))
        context = 3000 + int(rng.integers(0, 500))  # system prompt, tool list and the task
        prefix_id, prefix = "kk-system", 3000
        for _ in range(steps):
            out = int(lognormal(rng, 120, 0.7, 20, 500))
            rows.append(dict(arrival_s=t, customer="kodekart", slo_class="agent", input_tokens=context,
                             output_tokens=out, prefix_id=prefix_id, prefix_tokens=prefix))
            # The next step repeats this whole prompt, then adds the reply and a tool result.
            prefix_id, prefix = f"kk-task-{task:02d}", context
            context += out + int(lognormal(rng, 500, 0.8, 50, 3000))
            t += float(rng.lognormal(np.log(4), 0.6))  # the tool runs, the agent thinks
    return rows


def nyaya(rng, requests=100, contracts=15):
    docs = lognormal(rng, 35000, 0.35, 20000, 58000, contracts)  # each contract's length in tokens
    times = np.sort(rng.uniform(10 * 3600, 18 * 3600, requests))
    rows = []
    for t in times:
        d = int(rng.integers(0, contracts))
        question = int(lognormal(rng, 80, 0.5, 20, 400))
        rows.append(dict(arrival_s=t, customer="nyaya", slo_class="long", input_tokens=int(docs[d]) + question,
                         output_tokens=int(lognormal(rng, 200, 0.5, 40, 600)), prefix_id=f"ny-doc-{d:02d}",
                         prefix_tokens=int(docs[d])))
    return rows


def batch(rng, requests=800):
    """A nightly job: everything is submitted within two minutes, from 02:00."""
    times = np.sort(7200 + rng.uniform(0, 120, requests))
    inp = lognormal(rng, 1200, 0.5, 300, 4000, requests)
    out = lognormal(rng, 150, 0.4, 40, 400, requests)
    return [dict(arrival_s=t, customer="batch", slo_class="batch", input_tokens=int(i), output_tokens=int(o),
                 prefix_id="batch-system", prefix_tokens=150) for t, i, o in zip(times, inp, out)]


def write(name, rows):
    rows = sorted(rows, key=lambda r: r["arrival_s"])
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            r["arrival_s"] = round(float(r["arrival_s"]), 3)
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    sets = {"tidewater": tidewater(rng), "kodekart": kodekart(rng), "nyaya": nyaya(rng), "batch": batch(rng)}
    short = {"tidewater": "tw", "kodekart": "kk", "nyaya": "ny", "batch": "nb"}
    for name, rows in sets.items():
        rows.sort(key=lambda r: r["arrival_s"])
        for n, r in enumerate(rows, 1):
            r["id"] = f"{short[name]}-{n:05d}"
        # id first in each line, as the README lists the fields
        sets[name] = [{"id": r["id"], **{k: v for k, v in r.items() if k != "id"}} for r in rows]
        write(name, sets[name])
    write("mixed", [r for rows in sets.values() for r in rows])
    for name in [*sets, "mixed"]:
        print(f"{name:<10}{sum(1 for _ in open(OUT / f'{name}.jsonl')):>6} requests")


if __name__ == "__main__":
    main()
