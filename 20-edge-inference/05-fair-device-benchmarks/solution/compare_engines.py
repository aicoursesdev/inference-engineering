"""One model, three engines, same client, same prompt.

Needs: a GGUF file (set GGUF), Ollama running with `ollama pull qwen3:0.6b`,
and `uv run --with mlx-lm --with httpx python compare_engines.py`.
Engines run one at a time, in rounds, so slow moments hit all of them.
"""
import os
import statistics
import subprocess
import time

import httpx

from chat_bench import bench

GGUF = os.environ["GGUF"]            # the same bytes Ollama serves
LLAMA = os.environ.get("LLAMA_SERVER", "llama-server")
ROUNDS = 3


def wait(url):
    for _ in range(120):
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)  # still loading
    raise RuntimeError(f"{url} never came up")


def with_server(cmd, health, base, model):
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    try:
        wait(health)
        return bench(base, model)
    finally:
        p.terminate()
        p.wait()


ENGINES = {
    "llama-server": lambda: with_server(
        [LLAMA, "-m", GGUF, "--port", "8093", "-c", "4096"],
        "http://localhost:8093/health", "http://localhost:8093", "x"),
    "Ollama": lambda: bench("http://localhost:11434", "qwen3:0.6b"),
    "mlx_lm.server": lambda: with_server(
        ["mlx_lm.server", "--model", "mlx-community/Qwen3-0.6B-4bit",
         "--port", "8094"],
        "http://localhost:8094/v1/models", "http://localhost:8094",
        "mlx-community/Qwen3-0.6B-4bit"),
}

if __name__ == "__main__":
    print("load average before:", [round(x, 1) for x in os.getloadavg()])
    res = {k: [] for k in ENGINES}
    for _ in range(ROUNDS):
        for name, run in ENGINES.items():
            res[name].append(run())
    print(f"{'engine':14s} {'TTFT ms':>8s} {'tok/s':>7s}  per round")
    for name, rs in res.items():
        t = statistics.median(r[0] for r in rs) * 1000
        s = statistics.median(r[1] for r in rs)
        print(f"{name:14s} {t:8.0f} {s:7.0f}  "
              + " ".join(f"{r[1]:.0f}" for r in rs))
    print("load average after:", [round(x, 1) for x in os.getloadavg()])
