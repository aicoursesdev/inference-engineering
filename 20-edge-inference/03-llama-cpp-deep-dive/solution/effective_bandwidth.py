"""How much of the memory bandwidth does decoding really use?

Every generated token reads the whole file once, so
  effective GB/s = file size x tokens per second.
Reads the table sweep.py printed (results_ladder.txt).
"""
import sys

from servinglab.data import hardware

peak = hardware("M5-Max-32c")["bandwidth_gbs"]   # Apple's published 460 GB/s (32-core GPU)
print(f"{'file':24s} {'GiB':>5s} {'tok/s':>6s} {'GB/s':>6s} {'of peak':>8s}")
for line in open(sys.argv[1]).read().splitlines()[1:]:
    name, gib, _, tg = line.split()[:4]
    tg = float(tg.split("±")[0])
    gbs = float(gib) * 2**30 / 1e9 * tg
    print(f"{name[:-5]:24s} {gib:>5s} {tg:6.0f} {gbs:6.0f} {gbs / peak:8.0%}")
