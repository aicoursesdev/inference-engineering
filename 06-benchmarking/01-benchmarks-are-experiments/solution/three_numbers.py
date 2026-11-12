"""One server, four honest "tokens per second" numbers: only the workload and the load differ.

    llama-server -m models/Qwen3-0.6B-Q8_0.gguf --port 8080 -c 16384 -np 4
    uv run python three_numbers.py http://localhost:8080
"""

import asyncio
import sys

from servinglab import bench
from servinglab.data import traces


def fixed(n, tokens_in=128, tokens_out=128):
    """n identical requests: the benchmark everybody writes first."""
    row = {"id": "x", "slo_class": "interactive", "input_tokens": tokens_in, "output_tokens": tokens_out,
           "prefix_id": "x", "prefix_tokens": 0}
    return bench.make_requests([row] * n, max_input=tokens_in, max_output=tokens_out)


def measure(url, reqs, concurrency):
    """Output tokens per second when `concurrency` requests are always in flight."""
    results, seconds = asyncio.run(bench.run(url, "local", reqs, rate=float("inf"), max_concurrency=concurrency))
    ok = [r for r in results if r["ok"]]
    return sum(r["output_tokens"] for r in ok) / seconds, bench.percentile([r["ttft"] for r in ok], 50)


if __name__ == "__main__":
    url = sys.argv[1]
    tide = bench.make_requests(traces("tidewater")[:48], max_input=1500, max_output=128, share_prefix=False)
    kode = bench.make_requests(traces("kodekart")[:24], max_input=3500, max_output=128, share_prefix=False)
    cases = [("fixed 128-token prompt, 1 at a time", fixed(24), 1),
             ("fixed 128-token prompt, 4 at a time", fixed(48), 4),
             ("Tidewater lengths, 4 at a time", tide, 4),
             ("Kodekart 3,500-token prompts, 4 at a time", kode, 4)]
    measure(url, fixed(8), 4)  # warm up the server first, and throw the result away
    for name, reqs, c in cases:
        tps, ttft = measure(url, reqs, c)
        print(f"{name:<44} {tps:>6.0f} tokens/s   median TTFT {ttft * 1000:>5.0f} ms")
