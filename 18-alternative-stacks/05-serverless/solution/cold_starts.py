"""One serverless replica on a day of Tidewater-style traffic.

A container starts on the first request, stays warm until `window` seconds
after its last one, then scales to zero. The first request after a gap pays
the cold start. Inputs are ASSUMPTIONS (arrival rates, 60 s and 6 s cold
starts: Modal says vLLM start-up takes tens of seconds to minutes and
snapshots cut it "by a factor of 10"; 60 s is a stand-in inside that range).
shortcut: one container, no concurrency limit, requests take no time.
"""
import random

DAY = 86_400


def arrivals(seed, day_rate=0.02, night_rate=0.0005):
    """Poisson arrivals, req/s: busy 9:00-18:00 IST, near-silent otherwise."""
    rng, t, out = random.Random(seed), 0.0, []
    while t < DAY:
        rate = day_rate if 9 * 3600 <= t < 18 * 3600 else night_rate
        t += rng.expovariate(rate)
        out.append(t)
    return [x for x in out if x < DAY]


def simulate(times, window, cold_s):
    cold, warm_until, billed, start = 0, -1.0, 0.0, None
    for t in times:
        if t > warm_until:                      # scaled to zero: cold start
            if start is not None:
                billed += warm_until - start
            cold, start = cold + 1, t - cold_s  # billed while loading too
        warm_until = t + window
    if start is not None:
        billed += warm_until - start
    return cold, billed / 3600


times = arrivals(seed=3)
print(f"{len(times)} requests in a day; always-on = 24.0 GPU-hours\n")
print(f"{'cold start':>10}{'window':>8}{'cold requests':>15}{'GPU-hours billed':>18}")
for cold_s in (60, 6):
    for window in (15, 60, 300, 1800):
        c, h = simulate(times, window, cold_s)
        print(f"{cold_s:>8} s{window:>6} s{c:>9} ({c/len(times):4.0%}){h:>14.1f}")
