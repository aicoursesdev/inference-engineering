"""Turn a Module 06 benchmark file into dollars per million tokens.

    uv run python cost_from_bench.py bench-tidewater-4b-q4km.json 1.00

The second argument is the machine's price per hour. 1.00 here is a round
number for easy arithmetic, NOT a price anyone charges for this laptop.
"""
import json
import sys

from servinglab.cost import cost_per_mtok

data = json.load(open(sys.argv[1]))
rate = float(sys.argv[2])
print(f"{data['label']}, {data['machine']}, ${rate:.2f}/hour")
print("offered   output tok/s   $/M output   goodput/s   $/M output *useful*")
for run in data["runs"]:
    s = run["summary"]
    tps = s["output_tokens_per_s"]
    # useful output: only requests that met the SLO count
    ok = [r for r in run["results"] if r["ok"]]
    good_tokens = sum(r["output_tokens"] for r in ok if r["ttft"] <= s["slo"]["ttft"] and r["tpot"] <= s["slo"]["tpot"])
    useful_tps = good_tokens / s["seconds"]
    print(f"{s['offered_rate']:7g}   {tps:12.0f}   {cost_per_mtok(rate, tps):10.2f}   {s['goodput']:9.2f}   "
          f"{cost_per_mtok(rate, useful_tps) if useful_tps else float('inf'):10.2f}")
