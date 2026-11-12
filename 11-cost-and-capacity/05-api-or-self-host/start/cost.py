"""Lesson 11.5: your own breakeven.

The finished version is in servinglab/cost.py (breakeven.py uses that);
`make diff 11.5` shows what this file lacks.
"""

import math  # noqa: F401


def breakeven(monthly_fixed_usd, api_usd_per_request, self_usd_per_request=0.0):
    """Requests per month above which the fixed monthly bill beats paying an API per request.
    `self_usd_per_request` is any cost that still scales with traffic (usually ~0)."""
    raise NotImplementedError("TODO 11.5: fixed bill / (API price - your own per-request price), and math.inf when the API is no dearer, see the lesson's 'The break-even' section")
