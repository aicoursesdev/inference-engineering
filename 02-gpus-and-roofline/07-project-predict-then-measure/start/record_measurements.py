"""Measure your laptop with llama.cpp and save the numbers, with how they were made.

    uv run python record_measurements.py models/Qwen3-0.6B-Q8_0.gguf [more.gguf ...] --out mine.json

For each model it runs llama-bench for prefill (512 tokens) and decode (128 tokens), then
for the first model only: prefill at 1 to 2048 tokens, decode with 0, 4096 and 16384 tokens
already in the cache, and llama-batched-bench for 1 to 128 requests decoding together.
Close other programs first: a row whose runs disagree by more than 10% is flagged "noisy".
"""

import argparse
import json
import platform
import subprocess
import time


def run_json(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")] or json.loads(out)


def bench(model, *flags, repeats=5):
    rows = run_json(["llama-bench", "-m", model, "-r", str(repeats), "-o", "json", *flags])
    keep = ("model_type", "model_size", "model_n_params", "n_prompt", "n_gen", "n_depth", "avg_ts", "stddev_ts")
    return [{k: r.get(k) for k in keep} | {"noisy": r["stddev_ts"] > 0.1 * r["avg_ts"]} for r in rows]


def batched(model, parallel=(1, 2, 4, 8, 16, 32, 64, 128)):
    cmd = ["llama-batched-bench", "-m", model, "-c", "16384", "-npp", "128", "-ntg", "64",
           "-npl", ",".join(map(str, parallel)), "--output-format", "jsonl"]
    return [{k: r[k] for k in ("pl", "n_kv", "speed_pp", "speed_tg")} for r in run_json(cmd)]


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    both = subprocess.run(["llama-bench", "--version"], capture_output=True, text=True)
    version = [l for l in (both.stdout + both.stderr).splitlines() if l.startswith("version:")]
    return {"chip": chip.strip() or platform.processor(), "os": platform.platform(), "llama_cpp": version}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", default="measurements.json")
    a = ap.parse_args()
    first = a.models[0]
    data = {"recorded_at": time.strftime("%Y-%m-%d"), "machine": machine(),
            "per_model": {m: bench(m, "-p", "512", "-n", "128") for m in a.models},
            "prefill_by_length": bench(first, "-p", "1,2,4,8,16,32,64,128,256,512,1024,2048", "-n", "0", repeats=3),
            "decode_by_depth": bench(first, "-p", "0", "-n", "128", "-d", "0,4096,16384", repeats=3),
            "batched_decode": batched(first)}
    json.dump(data, open(a.out, "w"), indent=1)
    noisy = [r for rows in data["per_model"].values() for r in rows if r["noisy"]]
    print(f"wrote {a.out}" + (f"; {len(noisy)} noisy rows, rerun when the machine is quiet" if noisy else ""))
