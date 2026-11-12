"""One day of Monsoon capacity: on-demand floor plus spot, with interruptions.

Every input below is an ASSUMPTION for you to replace with your quotes.
The $2.00 GPU-hour is DeepSeek's own accounting figure (Open Infra Index);
the 40% spot price, 6% hourly interruption and 15-minute reload are made up
to show the method. shortcut: hours are independent, demand is a hand curve.
"""
import random

ON_DEMAND = 2.00            # $ per GPU-hour
SPOT = 0.40 * ON_DEMAND     # assumption
P_INTERRUPT = 0.06          # chance a spot node is reclaimed in a given hour
RELOAD_H = 0.25             # hours to get a replacement node serving again
GPUS_PER_NODE = 8
# nodes needed each hour: Tidewater by day, Nyaya and batch fill the night
DEMAND = [4] * 6 + [6, 8, 10, 12, 12, 12, 12, 12, 12, 11, 10, 9] + [8, 6, 5, 4, 4, 4]
assert len(DEMAND) == 24


def day(floor, seed):
    rng = random.Random(seed)
    cost = short = 0.0
    for need in DEMAND:
        spot_want = max(0, need - floor)
        lost = sum(rng.random() < P_INTERRUPT for _ in range(spot_want))
        cost += floor * GPUS_PER_NODE * ON_DEMAND            # reserved all day
        cost += spot_want * GPUS_PER_NODE * SPOT             # spot, billed up
        short += lost * RELOAD_H                              # node-hours missing
    return cost, short


peak_cost = max(DEMAND) * 24 * GPUS_PER_NODE * ON_DEMAND
need_cost = sum(DEMAND) * GPUS_PER_NODE * ON_DEMAND
print(f"peak-sized on-demand all day:   ${peak_cost:8,.0f}")
print(f"on-demand exactly as needed:    ${need_cost:8,.0f}  (if you could)")
for floor in (0, 4, 8):
    runs = [day(floor, s) for s in range(1000)]
    c = sum(r[0] for r in runs) / 1000
    s = sum(r[1] for r in runs) / 1000
    worst = sorted(r[1] for r in runs)[989]
    print(f"floor {floor:2} on-demand + spot: ${c:8,.0f}   "
          f"node-hours short: mean {s:4.1f}, p99 {worst:4.1f}")
