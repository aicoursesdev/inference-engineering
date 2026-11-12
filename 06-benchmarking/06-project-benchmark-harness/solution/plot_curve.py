"""Draw the latency-throughput curve of a saved sweep.

    uv sync --extra bench          # matplotlib, once
    uv run python plot_curve.py tidewater-sweep.json curve.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")  # draw to a file: no window needed
import matplotlib.pyplot as plt


def draw(data, path):
    runs = [r["summary"] for r in data["runs"]]
    rates = [s["offered_rate"] for s in runs]
    slo = data["slo"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for key, style in (("p50", "-"), ("p95", "--"), ("p99", ":")):
        left.plot(rates, [s["ttft"][key] for s in runs], style, marker="o", label=f"TTFT {key}")
    left.axhline(slo["ttft"], color="green", linewidth=1, label=f"target {slo['ttft']} s")
    left.set(xlabel="offered rate (requests/s)", ylabel="seconds", title="Time to first token", yscale="log")
    left.legend()
    right.plot(rates, [s["request_throughput"] for s in runs], marker="o", label="completed per second")
    right.plot(rates, [s["goodput"] for s in runs], marker="o", color="green", label="goodput (met the SLO)")
    right.plot(rates, rates, color="grey", linewidth=1, linestyle=":", label="offered")
    right.set(xlabel="offered rate (requests/s)", ylabel="requests/s", title="Throughput and goodput")
    right.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    draw(json.load(open(sys.argv[1])), sys.argv[2])
