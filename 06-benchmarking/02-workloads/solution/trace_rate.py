"""How Tidewater's real-looking day differs from a steady rate: requests per minute.

    uv run python trace_rate.py
"""

import numpy as np

from servinglab.data import traces


def per_minute(rows):
    """Requests arriving in each minute of the day (1,440 counts)."""
    return np.bincount([int(r["arrival_s"] // 60) for r in rows], minlength=1440)


if __name__ == "__main__":
    counts = per_minute(traces("tidewater"))
    day = counts[9 * 60:18 * 60]  # 09:00 to 18:00 IST
    spike = counts[12 * 60:12 * 60 + 30]
    print(f"09:00-18:00 average: {day.mean():.2f} requests a minute ({day.mean() / 60:.3f} a second)")
    print(f"busiest minute of the day: {counts.max()} requests, {counts.max() / day.mean():.1f} times the daytime average")
    print(f"12:00-12:30 sale spike: {spike.mean():.2f} a minute, {spike.mean() / day.mean():.1f} times the daytime average")
    print(f"quietest hour of the day has {counts.reshape(24, 60).sum(axis=1).min()} requests")
