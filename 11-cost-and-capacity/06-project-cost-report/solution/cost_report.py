"""The project's report: cost per million tokens, a capacity plan and a break-even, as Markdown.

    uv run python cost_report.py bench.json --rate 1.00 --prefill-tps 907 \
        --api-in 0.02 --api-out 0.04 --peak-qps 0.17 --mean-in 403 --mean-out 144

bench.json is a Module 06 `servinglab.bench --out` file. --rate is dollars per GPU-hour, --api-* are the
comparison API's dollars per million tokens: print where each came from beside the report. Output is
the arithmetic of servinglab.cost, nothing more.
"""
import argparse
import json

from servinglab.cost import breakeven, cost_per_mtok, gpus_needed, request_cost

ap = argparse.ArgumentParser()
ap.add_argument("bench")
ap.add_argument("--rate", type=float, required=True, help="dollars per GPU-hour")
ap.add_argument("--prefill-tps", type=float, required=True, help="input tokens/s you measured")
ap.add_argument("--api-in", type=float, required=True)
ap.add_argument("--api-out", type=float, required=True)
ap.add_argument("--peak-qps", type=float, required=True)
ap.add_argument("--mean-in", type=float, required=True)
ap.add_argument("--mean-out", type=float, required=True)
a = ap.parse_args()

data = json.load(open(a.bench))
# the capacity rate: the highest offered rate where 90% of requests met the SLO
ok_runs = [r for r in data["runs"] if r["summary"]["slo_attainment"] >= 0.9]
if not ok_runs:
    raise SystemExit("no rate met the SLO for 90% of requests: lower the rates and rerun the benchmark")
best = max(ok_runs, key=lambda r: r["summary"]["offered_rate"])["summary"]
decode_tps = best["output_tokens_per_s"]

req = request_cost(a.rate, a.prefill_tps, decode_tps, a.mean_in, a.mean_out)
plan = gpus_needed(a.peak_qps, a.mean_in, a.mean_out, a.prefill_tps, decode_tps)
api_per_request = (a.mean_in * a.api_in + a.mean_out * a.api_out) / 1e6
month = a.rate * 730

print(f"# Cost report: {data['label']}\n")
print(f"Benchmark {data['created']} on {data['machine']}; SLO TTFT {data['slo']['ttft']} s, TPOT {data['slo']['tpot'] * 1000:.0f} ms.")
print(f"Capacity at the SLO: {best['offered_rate']:g} requests/s, {decode_tps:.0f} output tokens/s.\n")
print(f"## Cost at ${a.rate:.2f} per GPU-hour")
print(f"- input ${req['input_per_mtok']:.3f} and output ${req['output_per_mtok']:.3f} per million tokens (full load)")
print(f"- output only: ${cost_per_mtok(a.rate, decode_tps):.3f}; at 30% utilisation ${cost_per_mtok(a.rate, decode_tps, 0.3):.3f}\n")
print(f"## Capacity for {a.peak_qps} requests/s peak")
print(f"- {plan['busy']:.2f} GPUs busy -> {plan['gpus']} with 30% headroom and 1 spare ({plan['utilisation']:.0%} busy at peak)\n")
print("## Versus the API")
print(f"- API ${api_per_request:.7f} per request; one GPU always on ${month:,.2f} a month")
print(f"- break-even {breakeven(month, api_per_request):,.0f} requests a month")
