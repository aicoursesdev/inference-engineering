"""Turn measured rows into cost per million output tokens, at matched latency.

Usage: python report.py results.csv [--tpot-ms 50]
Columns: stack,gpu,gpus,usd_per_gpu_hour,output_tok_s,ttft_p95_ms,tpot_p95_ms
Cost per million output tokens = (gpus x $/GPU-hour) / (3600 x tok/s / 1e6),
the formula InferenceX's posts use (their "TCO $/GPU/hr over tokens/s/GPU").
Rows that miss the TPOT limit are listed but not ranked: a cheap stack that
breaks the latency target is not cheaper, it is not an answer.
"""
import csv, sys


def usd_per_m(gpus, rate, tok_s):
    raise NotImplementedError("TODO 18.6: (gpus x rate) dollars an hour divided by millions of tokens an hour (3600 x tok_s / 1e6), see the lesson's 'Steps' section")


def rank(rows, tpot_ms):
    raise NotImplementedError("TODO 18.6: keep rows with tpot_p95_ms <= tpot_ms, sort them by cost per million output tokens, return (ranked, rows that missed, the cost function), see the lesson's 'Steps' section")


def self_test():
    # 1 GPU at $2/h making 1,000 tok/s: 3.6M tokens an hour -> $0.5556 per M
    assert round(usd_per_m(1, 2.0, 1000), 4) == 0.5556
    rows = [dict(stack="a", gpus="1", usd_per_gpu_hour="2", output_tok_s="1000",
                 tpot_p95_ms="40"),
            dict(stack="b", gpus="1", usd_per_gpu_hour="2", output_tok_s="3000",
                 tpot_p95_ms="80")]
    ranked, bad, _ = rank(rows, 50)
    assert [r["stack"] for r in ranked] == ["a"] and bad[0]["stack"] == "b"


if __name__ == "__main__":
    self_test()
    if len(sys.argv) < 2:
        sys.exit("usage: python report.py results.csv [--tpot-ms 50]")
    limit = float(sys.argv[sys.argv.index("--tpot-ms") + 1]) if "--tpot-ms" in sys.argv else 50
    ranked, bad, cost = rank(list(csv.DictReader(open(sys.argv[1]))), limit)
    for r in ranked:
        print(f"{r['stack']:14}${cost(r):6.3f} per M output tokens, "
              f"TPOT p95 {r['tpot_p95_ms']} ms")
    for r in bad:
        print(f"{r['stack']:14}misses TPOT {limit:g} ms ({r['tpot_p95_ms']} ms): not ranked")
