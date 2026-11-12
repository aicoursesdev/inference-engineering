"""Measure the mini server with servinglab.bench: a sweep of request rates.

    uv run python loadtest.py http://localhost:8000 --label continuous --out continuous.json

Requests are Tidewater's, shortened (at most 48 prompt tokens and 16 new
ones) so a NumPy engine on a laptop can finish them.
"""

import argparse
import json

import httpx

from servinglab import bench
from servinglab.data import traces

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--label", default="")
ap.add_argument("--rates", default="1,2,4,8")
ap.add_argument("--requests", type=int, default=16)
ap.add_argument("--out", default="")
a = ap.parse_args()

# one throw-away request, so the first measured one does not pay for waking the server
httpx.post(a.url + "/v1/completions", json={"prompt": "hello", "max_tokens": 4}, timeout=120)
reqs = bench.make_requests(traces("tidewater")[:a.requests], max_input=48, max_output=16,
                           share_prefix=False)
data = bench.sweep(a.url, "smollm2-135m", reqs, [float(r) for r in a.rates.split(",")],
                   {"ttft": 2.0, "tpot": 0.5}, label=a.label, engine="servinglab mini server")
print(bench.report(data))
if a.out:
    json.dump(data, open(a.out, "w"), indent=1)
