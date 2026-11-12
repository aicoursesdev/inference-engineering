"""Drill 1: how many GPUs? 25 requests a second at peak, 1,000 tokens in, 250 out,
Llama 3.1 8B in 16-bit on an H100 SXM. Speeds are the roofline's ceilings."""
from servinglab.cost import gpus_needed
from servinglab.roofline import predict

QPS, N_IN, N_OUT, BATCH = 25, 1000, 250, 32
p = predict("Llama-3.1-8B-Instruct", "H100-SXM", batch=BATCH,
            context=N_IN + N_OUT // 2, prompt=N_IN)
prefill_tps = p["prefill"]["tokens_per_s"]
decode_tps = p["decode"]["tokens_per_s"]


def gpus_at(scale):
    """gpus_needed's answer when the GPU runs at `scale` x the roofline ceiling."""
    raise NotImplementedError("TODO 14.4: busy = QPS x (N_IN/prefill + N_OUT/decode) at the scaled speeds, then headroom and a spare (gpus_needed does it), see the lesson's 'Drill 1: how many GPUs?' section")


if __name__ == "__main__":
    print(f"ceiling: prefill {prefill_tps:,.0f} tok/s, decode {decode_tps:,.0f} tok/s at batch {BATCH}")
    for label, scale in (("at the ceiling", 1.0), ("at half the ceiling", 0.5)):
        g = gpus_at(scale)
        print(f"{label:<20} busy {g['busy']:.2f} GPUs -> buy {g['gpus']} "
              f"(30% headroom, 1 spare), utilisation {g['utilisation']:.0%}")

    # Memory check: does a batch of 32 fit beside the weights?
    kv_per_req = p["model"]["kv_bytes_per_token_f16"] * (N_IN + N_OUT)
    free = 80e9 * 0.9 - p["weight_bytes"]
    print(f"weights {p['weight_bytes'] / 1e9:.1f} GB; one request's KV {kv_per_req / 1e6:.0f} MB; "
          f"{int(free // kv_per_req)} requests fit in {free / 1e9:.1f} GB (90% of 80 GB)")
