"""The project's report: goodput for each design, the break-even, and two published checks.

    uv run python goodput_report.py
"""
from servinglab.data import traces
from disagg import Cost, SLO, kv_bytes, run_colocated, run_disagg, report

HOUR = [r for r in traces("mixed") if 12 * 3600 <= r["arrival_s"] < 13 * 3600]  # the festival-sale hour
COST, CACHE, GPUS, IB = Cost(), 250_000, 8, 50e9
tight = lambda ms: {"interactive": (1.0, ms / 1e3), "agent": (2.0, ms / 1e3), "long": (8.0, 2 * ms / 1e3), "batch": (None, None)}


def designs(scale, slo):
    out = {"colocated, least loaded": report(run_colocated(HOUR, GPUS, COST, scale, router="least_loaded", cache_tokens=CACHE), slo),
           "colocated, cache aware": report(run_colocated(HOUR, GPUS, COST, scale, router="cache_aware", cache_tokens=CACHE), slo),
           "colocated, chunked prefill": report(run_colocated(HOUR, GPUS, COST, scale, policy="chunked", router="cache_aware", cache_tokens=CACHE), slo)}
    for p in (2, 3, 4):
        out[f"disaggregated {p}+{GPUS - p}, cache aware"] = report(
            run_disagg(HOUR, p, GPUS - p, COST, IB, scale, router="cache_aware", cache_tokens=CACHE), slo)
    return out


def table(title, rows):
    print(f"\n{title}")
    for name, r in rows.items():
        print(f"  {name:36} goodput {r['goodput_rps']:6.2f}/s  met {r['attained']*100:5.1f}%  "
              f"TTFT p95 {r['ttft_p95_s']:5.2f} s  TPOT p95 {r['tpot_p95_s']*1000:5.1f} ms")


def validate():
    print("\nTwo published checks (DistServe, Zhong et al., OSDI 2024):")
    # OPT-66B: 64 layers, hidden 9216, multi-head attention, FP16 (facebook/opt-66b config.json)
    b = kv_bytes(64, 9216, 512)
    print(f"  1. KV of one 512-token OPT-66B request: {b / 2**30:.3f} GiB (paper: about 1.13 GB)")
    print(f"     at 10 requests/s: {10 * b / 2**30:.2f} GiB/s = {10 * b * 8 / 2**30:.0f} Gbit/s (paper: 11.3 GB/s, 90 Gbps)")
    # OPT-175B: 96 layers, hidden 12288 (OPT paper, table 1). NVLink between A100s: 600 GB/s total, 300 GB/s one way.
    b175 = kv_bytes(96, 12288, 512)
    t = b175 / 300e9
    print(f"  2. KV of a 512-token OPT-175B request: {b175 / 1e9:.2f} GB; over 300 GB/s one-way NVLink: {t * 1e3:.1f} ms")
    print(f"     paper: over 95% of requests wait under 30 ms, and transfer is under 0.1% of latency")
    print(f"     for the 0.1% to hold, a request must last more than {t / 0.001:.0f} s (its TPOT target is 0.2 s: 40 tokens)")


if __name__ == "__main__":
    slo = SLO
    for scale in (240, 480):
        table(f"Monsoon's targets, festival hour sped up {scale}x, {GPUS} H100 GPUs, Llama 3.1 8B", designs(scale, slo))
    print("\nCapacity: the fastest festival hour each design serves with 95% of requests meeting both targets")
    for name, run in [("colocated, cache aware", lambda sc: run_colocated(HOUR, GPUS, COST, sc, router="cache_aware", cache_tokens=CACHE)),
                      ("colocated, chunked prefill", lambda sc: run_colocated(HOUR, GPUS, COST, sc, policy="chunked", router="cache_aware", cache_tokens=CACHE))] + [
            (f"disaggregated {p}+{GPUS - p}, cache aware", lambda sc, p=p: run_disagg(HOUR, p, GPUS - p, COST, IB, sc, router="cache_aware", cache_tokens=CACHE))
            for p in (2, 3, 4, 5)]:
        best = max([sc for sc in range(120, 1081, 120) if report(run(sc))["attained"] >= 0.95], default=None)
        print(f"  {name:36} " + (f"{best}x = {len(HOUR) * best / 3600:5.1f} requests/s offered" if best else "none"))
    print("\nLink speed, disaggregated 4+4, cache aware, 600x")
    for name, bw in [("NVLink 450 GB/s", 450e9), ("400 Gb/s NIC 50 GB/s", 50e9), ("100 Gb/s 12.5 GB/s", 12.5e9), ("25 Gb/s 3.1 GB/s", 3.125e9), ("10 Gb/s 1.25 GB/s", 1.25e9)]:
        r = report(run_disagg(HOUR, 4, 4, COST, bw, 600, router="cache_aware", cache_tokens=CACHE))
        print(f"  {name:22} met {r['attained']*100:5.1f}%  TTFT p95 {r['ttft_p95_s']:5.2f} s  TPOT p95 {r['tpot_p95_s']*1000:5.1f} ms")
    validate()
