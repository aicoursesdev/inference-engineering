"""Module 11: what a token costs, how many GPUs a load needs, and when your own beat an API.

    from servinglab.cost import cost_per_mtok, request_cost, gpus_needed, breakeven
    from servinglab.cost import hourly_rate, api_price

    cost_per_mtok(rate=2.69, tokens_per_s=1000)               # one GPU's rate, one measured speed
    request_cost(2.69, prefill_tps=20000, decode_tps=1000, input_tokens=350, output_tokens=120)

Every function takes numbers you measured or looked up, never a hidden default price.
`hourly_rate` and `api_price` look a price up in data/prices.json; each row carries the page
and the date it was read, so print both next to any figure you quote.

Throughput inputs mean what the Module 06 harness reports: `decode_tps` is the OUTPUT tokens per
second the GPU sustains at the batch you will really run (aggregate over all requests, not
one request's speed), `prefill_tps` is the INPUT tokens per second it processes.
"""

import math

from .data import prices as _prices


def hourly_rate(provider, gpu, tier=None):
    """(usd per hour, the price row) for a GPU at a provider; the row has the url and date."""
    rows = [r for r in _prices()["gpu_hourly"]
            if r["provider"] == provider and r["gpu"] == gpu and (tier is None or r.get("tier") == tier)]
    if not rows:
        raise ValueError(f"no price for {gpu} at {provider}")
    row = min(rows, key=lambda r: r["usd_per_hour"])
    return row["usd_per_hour"], row


def api_price(provider, model):
    """The (input, output) USD per million tokens for an API model, and its price row."""
    for r in _prices()["api_per_million_tokens"]:
        if r["provider"] == provider and r["model"] == model:
            return (r["input_usd_per_mtok"], r["output_usd_per_mtok"]), r
    raise ValueError(f"no API price for {model} at {provider}")


def cost_per_mtok(rate, tokens_per_s, utilisation=1.0):
    """USD per million tokens: the GPU's hourly rate over the tokens it really produces in an hour.
    `utilisation` is the share of the paid hour spent doing useful work (0 to 1)."""
    if not 0 < utilisation <= 1:
        raise ValueError("utilisation must be in (0, 1]")
    return rate / (tokens_per_s * 3600 * utilisation) * 1e6


def request_cost(rate, prefill_tps, decode_tps, input_tokens, output_tokens, utilisation=1.0):
    """What one request costs, input and output priced apart.
    Returns dict: usd (the request), input_per_mtok, output_per_mtok, gpu_seconds."""
    t_in, t_out = input_tokens / prefill_tps, output_tokens / decode_tps
    usd_s = rate / 3600 / utilisation
    return {"gpu_seconds": t_in + t_out, "usd": (t_in + t_out) * usd_s,
            "input_per_mtok": usd_s * t_in / input_tokens * 1e6,
            "output_per_mtok": usd_s * t_out / output_tokens * 1e6}


def gpus_needed(qps, input_tokens, output_tokens, prefill_tps, decode_tps, headroom=0.3, spares=1):
    """GPUs for a steady load. Busy GPU-seconds per second = qps x (prefill + decode seconds per request).
    `headroom` is the share of capacity kept free for bursts and the p99 (0.3 = run at 70%);
    `spares` extra GPUs survive one failure. Returns dict: busy, gpus, utilisation."""
    busy = qps * (input_tokens / prefill_tps + output_tokens / decode_tps)
    gpus = math.ceil(busy / (1 - headroom)) + spares
    return {"busy": busy, "gpus": gpus, "utilisation": busy / gpus}


def breakeven(monthly_fixed_usd, api_usd_per_request, self_usd_per_request=0.0):
    """Requests per month above which the fixed monthly bill for your own GPUs beats paying an API
    per request. `self_usd_per_request` is any cost that still scales with traffic (usually ~0).
    Valid only while the GPUs can carry that many requests: check against their capacity."""
    saving = api_usd_per_request - self_usd_per_request
    return math.inf if saving <= 0 else monthly_fixed_usd / saving
