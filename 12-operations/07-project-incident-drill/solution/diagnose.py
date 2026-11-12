"""The drill's evidence, one incident at a time. Run it, read it, then write each incident up.

    uv run python diagnose.py            # all three
    uv run python diagnose.py 2          # one

Incident 1 (latency): real llama-server runs. Incident 2 (out of memory): SIMULATED, see the
files' first lines. Incident 3 (silent quality): real runs of two model files on a 30-question canary.
"""
import json, re, statistics, sys
from pathlib import Path

DIR = Path(__file__).resolve().parents[3] / "data" / "recorded" / "incidents"
jsonl = lambda name: [json.loads(l) for l in (DIR / name).read_text().splitlines()]
med = lambda rows, k: statistics.median(r[k] for r in rows)


def p95(sorted_ms):
    """The 95th percentile of an ascending list, by the nearest-rank-below rule used in the lesson."""
    return sorted_ms[int(0.95 * (len(sorted_ms) - 1))]


def incident_1():
    print("INCIDENT 1: first tokens got slow after a deploy (real runs)")
    for run in ("good", "bad"):
        rows = jsonl(f"incident1-{run}.jsonl")
        t = sorted(r["ttft_ms"] for r in rows)
        print(f"  {run:4} TTFT p50 {t[len(t) // 2]:7.1f} ms  p95 {p95(t):7.1f} ms | prompt tokens computed {med(rows, 'prompt_n'):5.0f}"
              f" reused {med(rows, 'cache_n'):5.0f} | prefill {med(rows, 'prompt_ms'):6.1f} ms")
    for run in ("good", "bad"):
        m = dict(l.split() for l in (DIR / f"incident1-{run}.metrics.txt").read_text().splitlines() if not l.startswith("#"))
        print(f"  {run:4} /metrics: prompt_tokens_total {m['llamacpp:prompt_tokens_total']}, prompt_tokens_cached_total {m['llamacpp:prompt_tokens_cached_total']}")
    print("  (counters are cumulative: the bad file includes the good run, so subtract)")


def incident_2():
    print("INCIDENT 2: the engine died at 02:00:10 (SIMULATED)")
    rows = jsonl("incident2-metrics.jsonl")
    for r in rows[-4:]:
        print(f"  {r['t']} running {r['num_requests_running']:3}  kv_cache_usage {r['kv_cache_usage_perc'] * 100:5.1f}%  "
              f"free GPU memory {r['gpu_free_gb'] * 1000:6.0f} MB  neighbour process {r['sidecar_gb']:.2f} GB")
    print("  " + (DIR / "incident2-server.log").read_text().splitlines()[-2][:200])


def incident_3():
    print("INCIDENT 3: nothing alarmed, users say answers are worse (real runs)")
    for name in ("incident3-good-Q8_0.json", "incident3-bad-Q2_K.json"):
        d = json.loads((DIR / name).read_text())
        stray = sum(bool(re.search(r"[؀-ۿ฀-๿぀-ヿ一-鿿]", r["reply"])) for r in d["results"])
        print(f"  {name:28} canary {d['score']}/{d['total']}, replies with stray script: {stray}")


if __name__ == "__main__":
    pick = sys.argv[1:] or ["1", "2", "3"]
    for n in pick:
        {"1": incident_1, "2": incident_2, "3": incident_3}[n]()
        print()
