"""The same average rate, three different traffic shapes.

    uv run python arrivals.py
"""

import numpy as np

from servinglab.bench import arrival_times


def busiest_second(times):
    """The most requests that arrived within any one wall-clock second."""
    return int(np.bincount(times.astype(int)).max())


if __name__ == "__main__":
    rate, n = 8, 2400  # 8 requests a second for five minutes
    print(f"{'burstiness':>10} {'mean gap':>9} {'busiest second':>15} {'longest quiet':>14}")
    for b in (4.0, 1.0, 0.25):
        t = arrival_times(rate, n, burstiness=b, seed=0)
        gaps = np.diff(t)
        print(f"{b:>10} {gaps.mean() * 1000:>7.0f}ms {busiest_second(t):>13}/s {gaps.max():>12.1f}s")
