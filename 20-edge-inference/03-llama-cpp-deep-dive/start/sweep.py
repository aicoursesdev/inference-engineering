"""Run llama-bench over GGUF files and print one table.

python sweep.py model1.gguf model2.gguf ...   (extra flags after --)
Reads llama-bench's JSON. Reports the best of 5 runs: on a laptop that
is doing other things, noise only ever slows a run down.
"""
import json
import subprocess
import sys


def bench(path, *flags):
    cmd = ["llama-bench", "-m", path, "-p", "512", "-n", "128", "-r", "5",
           "-o", "json", *flags]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = json.loads(out.stdout)
    pp = next(r for r in rows if r["n_prompt"] > 0)
    tg = next(r for r in rows if r["n_gen"] > 0)
    return {"file": path.split("/")[-1], "gib": pp["model_size"] / 2**30,
            "pp512": max(pp["samples_ts"]), "tg128": max(tg["samples_ts"]),
            "tg_sd": tg["stddev_ts"]}


if __name__ == "__main__":
    args = sys.argv[1:]
    flags = args[args.index("--") + 1:] if "--" in args else []
    files = args[: args.index("--")] if "--" in args else args
    print(f"{'file':26s} {'GiB':>5s} {'pp512':>8s} {'tg128':>8s}")
    for f in files:
        r = bench(f, *flags)
        print(f"{r['file']:26s} {r['gib']:5.2f} {r['pp512']:8.0f} "
              f"{r['tg128']:6.0f}±{r['tg_sd']:.0f}")
