"""Check every file in data/: schemas, units, sources and traces.

    uv run python scripts/check_data.py

Exits 1 and lists what is wrong; prints a short summary when all is well.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
problems = []


def check(ok, message):
    if not ok:
        problems.append(message)


def is_source(s):
    return (isinstance(s, dict) and str(s.get("url", "")).startswith("https://")
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(s.get("fetched", ""))) is not None)


# ---- models ---------------------------------------------------------------
print("models/")
sources = json.loads((DATA / "models" / "sources.json").read_text())
files = sorted(p for p in (DATA / "models").glob("*.json") if p.stem != "sources")
check({p.stem for p in files} == set(sources), "models: sources.json and the config files differ")
for p in files:
    c = json.loads(p.read_text())
    check(is_source(sources.get(p.stem)), f"{p.name}: no source url and date in sources.json")
    for key in ("num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "vocab_size"):
        check(isinstance(c.get(key), int) and c[key] > 0, f"{p.name}: {key} missing")
    check(c["hidden_size"] % c["num_attention_heads"] == 0 or "head_dim" in c, f"{p.name}: head size unclear")
    check(c["num_attention_heads"] % c["num_key_value_heads"] == 0, f"{p.name}: heads not a multiple of KV heads")
print(f"  {len(files)} configs")

# ---- hardware -------------------------------------------------------------
print("hardware.json")
hw = json.loads((DATA / "hardware.json").read_text())["hardware"]
NUM = ("memory_gb", "bandwidth_gbs", "fp16_tflops", "fp8_tflops", "fp4_tflops")
check(len({h["name"] for h in hw}) == len(hw), "hardware: duplicate names")
for h in hw:
    n = h["name"]
    check(h["kind"] in ("gpu", "laptop"), f"{n}: kind {h['kind']}")
    for f in NUM:
        v = h[f]
        if v is None:
            continue
        check(isinstance(v, (int, float)) and v > 0, f"{n}: {f} is {v!r}")
        check(is_source(h["sources"].get(f)), f"{n}: {f} has a value but no source url and date")
        if not is_source(h["sources"].get(f)):
            continue
    for f in h["sources"]:
        check(f in NUM and h[f] is not None, f"{n}: source for {f}, which has no value")
    check(h["memory_gb"] is not None, f"{n}: no memory")
    nulls = [f for f in NUM if h[f] is None]
    check(not nulls or h["notes"], f"{n}: {nulls} are null but there is no note saying why")
    fp = [h[f] for f in ("fp16_tflops", "fp8_tflops", "fp4_tflops") if h[f] is not None]
    check(fp == sorted(fp), f"{n}: lower precision is not faster")
    if h["bandwidth_gbs"] and h["memory_gb"] and h["kind"] == "gpu":
        check(h["bandwidth_gbs"] / h["memory_gb"] < 100, f"{n}: bandwidth/memory looks off (units?)")
print(f"  {len(hw)} chips, {sum(any(s.get('derived') for s in h['sources'].values()) for h in hw)} with derived figures")

# ---- prices ---------------------------------------------------------------
print("prices.json")
pr = json.loads((DATA / "prices.json").read_text())
for r in pr["gpu_hourly"]:
    check(is_source({"url": r["url"], "fetched": r["fetched"]}) and r["usd_per_hour"] > 0, f"price {r['provider']} {r['gpu']}")
    if "usd_per_second" in r:
        check(abs(r["usd_per_second"] * 3600 - r["usd_per_hour"]) < 1e-3, f"price {r['provider']} {r['gpu']}: hour != second x 3600")
for r in pr["api_per_million_tokens"]:
    check(is_source({"url": r["url"], "fetched": r["fetched"]}), f"api price {r['model']}: source")
    check(0 < r["input_usd_per_mtok"] <= r["output_usd_per_mtok"] * 5, f"api price {r['model']}: values")
print(f"  {len(pr['gpu_hourly'])} GPU rates, {len(pr['api_per_million_tokens'])} API prices")

# ---- traces ---------------------------------------------------------------
print("traces/")
FIELDS = ["id", "arrival_s", "customer", "slo_class", "input_tokens", "output_tokens", "prefix_id", "prefix_tokens"]
SLO = {"tidewater": "interactive", "kodekart": "agent", "nyaya": "long", "batch": "batch"}
allrows = {}
for name in [*SLO, "mixed"]:
    rows = [json.loads(line) for line in open(DATA / "traces" / f"{name}.jsonl")]
    allrows[name] = rows
    check(rows, f"{name}: empty")
    check(len({r["id"] for r in rows}) == len(rows), f"{name}: duplicate ids")
    check([r["arrival_s"] for r in rows] == sorted(r["arrival_s"] for r in rows), f"{name}: not in time order")
    for r in rows:
        check(list(r) == FIELDS, f"{name} {r.get('id')}: fields {list(r)}")
        check(0 <= r["arrival_s"] < 86400, f"{r['id']}: arrival_s")
        check(r["slo_class"] == SLO.get(r["customer"]), f"{r['id']}: slo_class {r['slo_class']} for {r['customer']}")
        check(r["output_tokens"] > 0 and 0 < r["prefix_tokens"] <= r["input_tokens"], f"{r['id']}: token counts")
        if name != "mixed":
            check(r["customer"] == name, f"{r['id']}: customer {r['customer']} in {name}")
    print(f"  {name:<10}{len(rows):>5} requests, mean input {sum(r['input_tokens'] for r in rows) // len(rows):>6}")
check(sorted(r["id"] for r in allrows["mixed"]) == sorted(r["id"] for n in SLO for r in allrows[n]),
      "mixed.jsonl is not the four traces together")
nyaya = allrows["nyaya"]
check(all(20000 <= r["input_tokens"] <= 60500 for r in nyaya), "nyaya: prompts outside 20k-60k tokens")
by_doc = {}
for r in nyaya:
    by_doc.setdefault(r["prefix_id"], set()).add(r["prefix_tokens"])
check(all(len(v) == 1 for v in by_doc.values()), "nyaya: a document has two lengths")
check(Counter(r["prefix_id"] for r in allrows["kodekart"])["kk-system"] >= 5, "kodekart: tasks do not share the system prompt")
tw = allrows["tidewater"]
busy = sum(9 * 3600 <= r["arrival_s"] < 18 * 3600 for r in tw)
check(busy > 0.7 * len(tw), "tidewater: not mostly business hours")

# ---- recorded -------------------------------------------------------------
print("recorded/")
meta = json.loads((DATA / "recorded" / "llama-server-reply.json").read_text())
for key in ("machine", "llama_cpp", "model", "recorded_at", "request", "events"):
    check(key in meta, f"recorded: {key} missing")
check(meta["events"] and all("t_ms" in e and "data" in e for e in meta["events"]), "recorded: events need t_ms and data")
check([e["t_ms"] for e in meta["events"]] == sorted(e["t_ms"] for e in meta["events"]), "recorded: events out of order")
check(meta["request"].get("temperature") == 0, "recorded: not at temperature 0")
print(f"  {len(meta['events'])} events from {meta['machine']['chip']}")

if problems:
    print(f"\n{len(problems)} problem(s):")
    print("\n".join(f"  - {p}" for p in problems[:40]))
    sys.exit(1)
print("\nAll data files pass.")
