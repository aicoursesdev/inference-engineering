"""Does the speed hold? Repeat the same request for a few minutes.

llama-server -m model.gguf --port 8093
uv run --with httpx python sustained.py http://localhost:8093 x 120
A phone or a fanless board slows itself when it gets hot; a request-by-
request log is how you see it. First number is the cool-start speed.
"""
import sys
import time

from chat_bench import one

base, model, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
one(base, model, 8)  # warm-up
start = time.perf_counter()
window, rates = 20, []
print("seconds  decode tok/s (mean of the requests in each 20 s window)")
while time.perf_counter() - start < seconds:
    _, tps, _ = one(base, model, 256)
    rates.append(tps)
    t = time.perf_counter() - start
    if t >= window:
        print(f"{window - 20:3d}-{window:<3d}  {sum(rates) / len(rates):7.0f}",
              flush=True)
        window, rates = window + 20, []
