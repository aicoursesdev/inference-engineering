"""Size a prefill/decode fleet from published per-node throughput.

Per-node figures are LMSYS's (96 H100s, DeepSeek-V3, 2,000-token inputs):
52.3k input and 22.3k output tokens/s per node. They are THEIR setup;
replace them with your own measurement before you trust the answer.
"""
import math, sys

PREFILL_TPS, DECODE_TPS = 52_300, 22_300     # per 8-GPU node
GPUS, RATE = 8, 2.00                         # $ per GPU-hour (DeepSeek's figure)


def design(qps, in_tok, out_tok, cache_hit, headroom=0.25):
    new_in = qps * in_tok * (1 - cache_hit)  # prompt tokens that need prefill
    out = qps * out_tok
    p = math.ceil(new_in * (1 + headroom) / PREFILL_TPS)
    d = math.ceil(out * (1 + headroom) / DECODE_TPS)
    cost_hr = (p + d) * GPUS * RATE
    per_m_out = cost_hr / (out * 3600 / 1e6)
    return dict(prefill_nodes=p, decode_nodes=d, nodes=p + d,
                usd_per_hour=cost_hr, usd_per_m_output=per_m_out,
                prefill_load=new_in / (p * PREFILL_TPS),
                decode_load=out / (d * DECODE_TPS))


def self_check():
    # LMSYS says about $0.20 per million output tokens: one decode node, $2/GPU-hour
    one_node = GPUS * RATE / (DECODE_TPS * 3600 / 1e6)
    assert abs(one_node - 0.20) < 0.005, one_node
    # DeepSeek's page: 226.75 nodes x 8 GPUs x $2 x 24 h = $87,072
    assert round(226.75 * 8 * 2 * 24) == 87_072


if __name__ == "__main__":
    self_check()
    qps = float(sys.argv[1]) if len(sys.argv) > 1 else 200
    r = design(qps, in_tok=2000, out_tok=400, cache_hit=0.5)
    print(f"{qps:g} requests/s, 2,000 in (half cached), 400 out")
    for k, v in r.items():
        print(f"  {k:17}{v:12,.2f}" if isinstance(v, float) else f"  {k:17}{v:12}")
    seqs = 256 * GPUS
    print(f"\nLMSYS decode run: {DECODE_TPS:,} tok/s over {seqs} sequences "
          f"(if 256 is per GPU) = {DECODE_TPS/seqs:.1f} tok/s each,")
    print(f"i.e. {1000*seqs/DECODE_TPS:.0f} ms per token. Monsoon's agent class "
          f"allows 50 ms: that needs a smaller batch, so fewer tokens per node.")
