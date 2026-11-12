"""Lesson 11.4: your own gpus_needed.

The finished version is in servinglab/cost.py (size_fleet.py uses that);
`make diff 11.4` shows what this file lacks.
"""

import math  # noqa: F401


def gpus_needed(qps, input_tokens, output_tokens, prefill_tps, decode_tps, headroom=0.3, spares=1):
    """GPUs for a steady load. `headroom` is the share of capacity kept free (0.3 = run at 70%);
    `spares` extra GPUs survive one failure. Returns dict: busy, gpus, utilisation."""
    raise NotImplementedError("TODO 11.4: busy = qps x (input/prefill_tps + output/decode_tps); gpus = ceil(busy / (1 - headroom)) + spares; utilisation = busy / gpus, see the lesson's 'Headroom, and a spare' section")
