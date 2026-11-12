"""When does a rented L4 beat DeepInfra's Llama 3.1 8B API for Tidewater's average request?

The two are not the same model or quality; this is the money half of the decision only.
"""
from servinglab.cost import api_price, breakeven, hourly_rate
from servinglab.data import traces

MEAN_IN, MEAN_OUT = 403, 144  # Tidewater's trace means (see 04-capacity-worksheet)
rows = traces("tidewater")
assert round(sum(r["input_tokens"] for r in rows) / len(rows)) == MEAN_IN

(p_in, p_out), api = api_price("DeepInfra", "Meta-Llama-3.1-8B-Instruct-Turbo")
rate, gpu = hourly_rate("RunPod", "L4", "secure cloud")
per_request = (MEAN_IN * p_in + MEAN_OUT * p_out) / 1e6
month = rate * 730  # hours in an average month, always on
n = breakeven(month, per_request)
print(f"API: ${per_request:.7f} per request ({api['provider']}, read {api['fetched']})")
print(f"L4 always on: ${rate}/h x 730 h = ${month:,.2f} a month ({gpu['provider']} {gpu['tier']}, read {gpu['fetched']})")
print(f"break-even: {n:,.0f} requests a month = {n / (730 * 3600):.1f} requests per second, all day")
print(f"Tidewater sends {len(rows) * 30:,} a month")

# Can the L4 even carry that? Module 02's ceiling for Llama 3.1 8B in 16-bit, 32 at once.
from servinglab.roofline import predict

d = predict("Llama-3.1-8B-Instruct", "L4", batch=32, context=1000)["decode"]["tokens_per_s"]
print(f"an L4's decode ceiling is {d:,.0f} output tokens/s = {d / MEAN_OUT:.1f} requests/s at {MEAN_OUT} tokens each")
