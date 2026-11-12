"""Decode ceilings against llama-bench measurements, then one fitted number that explains the gap.

    uv run python ceiling.py [measurements.json]

The default file is data/recorded/llama-bench-m5max.json, measured on an Apple M5 Max laptop.
"""

import json
import re
import sys
from pathlib import Path

from servinglab.data import DATA
from servinglab.roofline import predict

CONTEXT = 64  # llama-bench's 128 generated tokens start from an empty cache: 64 on average


def load(path=DATA / "recorded" / "llama-bench-m5max.json"):
    data = json.loads(Path(path).read_text())
    rows = []
    for name, results in data["per_model"].items():
        decode = next(r for r in results if r["n_gen"] > 0)
        size = re.search(r"Qwen3-[\d.]+B", name).group()  # the config in data/models/
        rows.append({"file": Path(name).name, "model": size, "bytes": decode["model_size"],
                     "params": decode["model_n_params"], "measured": decode["avg_ts"]})
    return rows


def ceiling(row, bandwidth_gbs, overhead_seconds=0.0):
    """Predicted decode tokens per second for one llama-bench row."""
    raise NotImplementedError(
        "TODO 02.4: use predict with the row's bytes per parameter and return the decode tokens per second, see the lesson's 'The ceiling' section")


def fit_overhead(row, bandwidth_gbs):
    """The seconds per step the pure ceiling leaves unexplained for this one measurement."""
    raise NotImplementedError(
        "TODO 02.4: return the seconds per step that the pure ceiling leaves unexplained: 1 / measured minus 1 / ceiling, see the lesson's 'The gap, and one number that explains it' section")


if __name__ == "__main__":
    rows = load(*sys.argv[1:])
    bw = 406  # GB/s measured on this laptop by machine.py; Apple lists 460
    print(f"{'file':<24}{'MB':>6}{'ceiling':>9}{'measured':>10}{'of ceiling':>11}")
    for r in rows:
        c = ceiling(r, bw)
        print(f"{r['file']:<24}{r['bytes'] / 1e6:>6.0f}{c:>9.0f}{r['measured']:>10.0f}{r['measured'] / c:>11.0%}")
    anchor = next(r for r in rows if "Q8_0" in r["file"] and "0.6B" in r["file"])
    extra = fit_overhead(anchor, bw)
    print(f"\nFit one number from {anchor['file']} alone: {extra * 1000:.2f} ms of overhead per step.")
    print(f"{'file':<24}{'predicted':>10}{'measured':>10}{'error':>8}")
    for r in rows:
        p = ceiling(r, bw, extra)
        print(f"{r['file']:<24}{p:>10.0f}{r['measured']:>10.0f}{p / r['measured'] - 1:>+8.0%}")
