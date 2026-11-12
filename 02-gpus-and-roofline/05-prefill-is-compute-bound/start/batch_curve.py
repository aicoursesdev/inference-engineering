"""How the batch moves a step across the ridge: predicted against measured, on the laptop.

    uv run python batch_curve.py [measurements.json]

The roof is the one machine.py measured on the M5 Max laptop: 406 GB/s and 48.4 TFLOPS in float16.
"""

import json
import sys
from pathlib import Path

from servinglab.data import DATA
from servinglab.roofline import predict

CHIP = {"bandwidth_gbs": 406, "fp16_tflops": 48.4}
BYTES_PER_PARAM = 633_495_552 / 596_049_920  # the Q8_0 file
OVERHEAD = 0.00135  # seconds a step loses to fixed costs, fitted in lesson 02.4


def at(**kw):
    return predict("Qwen3-0.6B", CHIP, weight_bytes_per_param=BYTES_PER_PARAM, overhead_seconds=OVERHEAD, **kw)


def ridge_batch(bytes_per_param=BYTES_PER_PARAM, chip=CHIP):
    """The batch at which decode stops being memory-bound: ridge FLOPs/byte times bytes/param over 2."""
    raise NotImplementedError(
        "TODO 02.5: return the chip's ridge in FLOPs per byte times bytes per parameter over 2, see the lesson's 'Decode with a crowd: the batch dial' section")


if __name__ == "__main__":
    data = json.loads(Path(*sys.argv[1:] or [DATA / "recorded" / "llama-bench-m5max.json"]).read_text())
    print(f"ridge for this chip: decode turns compute-bound at a batch of about {ridge_batch():.0f}\n")
    print("Reading a prompt of n tokens (prefill), tokens per second")
    print(f"{'n':>6}{'predicted':>11}{'measured':>10}{'bound':>9}")
    for r in data["prefill_by_length"]:
        n = r["n_prompt"]
        p = at(prompt=n, context=n)["prefill"]
        print(f"{n:>6}{p['tokens_per_s']:>11.0f}{r['avg_ts']:>10.0f}{p['bound']:>9}")
    print("\nDecoding B requests together, total tokens per second")
    print(f"{'B':>6}{'predicted':>11}{'measured':>10}{'bound':>9}")
    for r in data["batched_decode"]:
        b = r["pl"]
        p = at(batch=b, context=160)["decode"]
        print(f"{b:>6}{p['tokens_per_s']:>11.0f}{r['speed_tg']:>10.0f}{p['bound']:>9}")
