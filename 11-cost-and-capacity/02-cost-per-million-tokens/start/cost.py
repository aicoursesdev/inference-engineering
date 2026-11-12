"""Lesson 11.2: your own cost_per_mtok and request_cost.

The finished versions are in servinglab/cost.py (the scripts here use those);
`make diff 11.2` shows what this file lacks.
"""


def cost_per_mtok(rate, tokens_per_s, utilisation=1.0):
    """USD per million tokens: the GPU's hourly rate over the tokens it really produces in an hour.
    `utilisation` is the share of the paid hour spent doing useful work (0 to 1, else ValueError)."""
    raise NotImplementedError("TODO 11.2: rate / (tokens_per_s * 3600 * utilisation) * 1e6, refusing a utilisation outside (0, 1], see the lesson's 'The formula, and a check' section")


def request_cost(rate, prefill_tps, decode_tps, input_tokens, output_tokens, utilisation=1.0):
    """What one request costs, input and output priced apart.
    Returns dict: usd (the request), input_per_mtok, output_per_mtok, gpu_seconds."""
    raise NotImplementedError("TODO 11.2: time the input at prefill_tps and the output at decode_tps, price each second, see the lesson's 'Input and output are different jobs' section")
