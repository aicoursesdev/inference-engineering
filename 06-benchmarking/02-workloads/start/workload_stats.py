"""Length distributions of Monsoon's four traces: what a benchmark must reproduce.

    uv run python workload_stats.py
"""

import numpy as np

from servinglab.data import traces


def summary(values):
    """Median, 95th and 99th percentile and the maximum of a list of lengths."""
    v = np.asarray(values)
    return {"median": int(np.median(v)), "p95": int(np.percentile(v, 95)),
            "p99": int(np.percentile(v, 99)), "max": int(v.max())}


if __name__ == "__main__":
    print(f"{'trace':<10}{'requests':>9}   {'input tokens: median p95 p99 max':<38}output tokens")
    for name in ("tidewater", "kodekart", "nyaya", "batch"):
        rows = traces(name)
        i, o = summary([r["input_tokens"] for r in rows]), summary([r["output_tokens"] for r in rows])
        print(f"{name:<10}{len(rows):>9}   {i['median']:>8}{i['p95']:>7}{i['p99']:>7}{i['max']:>7}"
              f"{'':<9}{o['median']:>4}{o['p95']:>6}{o['p99']:>6}{o['max']:>6}")
