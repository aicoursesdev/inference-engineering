"""Measure one on-device assistant on this machine, one GGUF level at a time.

python ondevice_report.py cpu-4threads Q2_K Q4_K_M Q8_0
Uses llama-server only (no other engine), so what changes is the level.
Writes one row per level to the ledger and prints a table.
"""
import os
import subprocess
import sys
import time

import httpx

from chat_bench import bench
from quality_probe import score
from servinglab.ledger import record

LLAMA = os.environ.get("LLAMA_SERVER", "llama-server")
MODELS = os.environ.get("MODELS", "models")
PORT = 8097
# Device settings: a name and the llama-server flags that make it that device.
DEVICES = {
    "gpu": ["-ngl", "99"],
    "cpu-4threads": ["-ngl", "0", "-t", "4"],   # stands in for a small board
}


def resident_mb(pid):
    kb = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                        capture_output=True, text=True).stdout
    return int(kb) / 1024


def ship_level(rows, memory_mb, slack=1, memory_share=0.6):
    """The ship rule: the fastest level whose quality is within `slack`
    questions of the best and whose resident memory is under `memory_share`
    of the device's memory. rows: (level, file MB, RSS MB, TTFT ms, tok/s,
    right, total) as one_level returns them, with the level first. None if no
    level qualifies."""
    raise NotImplementedError("TODO 20.6: keep levels within `slack` questions of the best and under `memory_share` of the device memory, return the fastest one's name or None, see the lesson's 'Build it' section")


def one_level(device, level):
    path = f"{MODELS}/Qwen3-0.6B-{level}.gguf"
    cmd = [LLAMA, "-m", path, "--port", str(PORT), "-c", "4096", "-np", "1", "--jinja",
           *DEVICES[device]]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                if httpx.get(f"http://localhost:{PORT}/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)  # still loading
        ttft, tps, _ = bench(f"http://localhost:{PORT}", "m", runs=3)
        right, total = score(f"http://localhost:{PORT}")
        mem = resident_mb(p.pid)
    finally:
        p.terminate()
        p.wait()
    size_mb = os.path.getsize(path) / 1e6
    record({"lesson": "20.6", "engine": "llama.cpp", "model": f"Qwen3-0.6B-{level}",
            "hardware": f"M5-Max/{device}", "change": f"level {level}",
            "ttft_ms": round(ttft * 1000), "tok_per_s": round(tps),
            "note": f"file {size_mb:.0f} MB; resident {mem:.0f} MB; "
                    f"quality {right}/{total}; load {os.getloadavg()[0]:.1f}"})
    return size_mb, mem, ttft * 1000, tps, right, total


if __name__ == "__main__":
    device, levels = sys.argv[1], sys.argv[2:]
    print(f"{device}: {'level':8s} {'file MB':>8s} {'RSS MB':>7s} "
          f"{'TTFT ms':>8s} {'tok/s':>6s} {'quality':>8s}")
    for level in levels:
        s, m, t, v, r, n = one_level(device, level)
        print(f"{'':{len(device)}}  {level:8s} {s:8.0f} {m:7.0f} "
              f"{t:8.0f} {v:6.0f} {r:5d}/{n}")
