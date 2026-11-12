"""Make the drill's incident 2: an out-of-memory crash. SIMULATED, not recorded from a GPU.

Run:  uv run python simulate_oom_incident.py   (writes ../../../data/recorded/incidents/incident2-*)

The story: Monsoon's L4 (24 GB) serves Qwen3-4B in bf16 with vLLM sized at
gpu_memory_utilization 0.90. To save money Kabir put the embeddings job's small model on the same
GPU. At 02:00 the nightly batch trace arrives (data/traces/batch.jsonl, 800 requests in two
minutes). The server survives until the decode batch is big enough that the sampler's temporary
buffer cannot get memory the sidecar's caching allocator already holds. Every size is computed
below from the real model config; the arrival times are the real trace; the service times and
the sidecar's growth are ASSUMPTIONS, stated here and in the log header. The log line formats
follow PyTorch's out-of-memory message, but this file was written by this script, not by an engine.
"""
import json
from pathlib import Path
from servinglab.data import traces, model_config, hardware

OUT = Path(__file__).resolve().parents[3] / "data" / "recorded" / "incidents"
GB = 1e9
cfg = model_config("Qwen3-4B")
H, L, I, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["intermediate_size"], cfg["vocab_size"]
q_dim, kv_dim = cfg["num_attention_heads"] * cfg["head_dim"], cfg["num_key_value_heads"] * cfg["head_dim"]
params = V * H + L * (H * q_dim + 2 * H * kv_dim + q_dim * H + 3 * H * I + 2 * H) + H  # tied embeddings, norms included
weights_gb = params * 2 / GB
kv_per_token = 2 * L * kv_dim * 2                      # K and V, bf16
total_gb = hardware("L4")["memory_gb"]
UTIL = 0.90
budget_gb = UTIL * total_gb
activation_gb = 1.6                                    # ASSUMPTION: vLLM's startup profile of the largest step
pool_gb = budget_gb - weights_gb - activation_gb
pool_tokens = int(pool_gb * GB / kv_per_token)
worker_ctx_gb = 0.5                                    # ASSUMPTION: CUDA context outside the budget
sidecar_steady_gb = 0.6e9 * 2 / GB + 0.5               # 0.6B-parameter embedding model in fp16 + its own context
sidecar_peak_growth_gb = 0.18                          # ASSUMPTION: the allocator's cache grows while the batch runs
logit_row_bytes = V * 4                                # one fp32 logits row per running sequence (the sampler's input)
free_idle = total_gb - budget_gb - worker_ctx_gb - sidecar_steady_gb


def run():
    reqs = [r for r in traces("batch")]
    t0 = 7200 - 60                                     # start the record a minute before the job
    events = []                                        # (finish time, tokens held)
    metrics, log = [], []
    crashed = None
    for second in range(0, 180, 5):
        now = t0 + second
        live = [(f, tok) for f, tok in events if f > now]
        running = len(live)
        used_tokens = sum(tok for _, tok in live)
        arrivals = [r for r in reqs if t0 + second - 5 <= r["arrival_s"] < now]
        for r in arrivals:                             # ASSUMPTION: 25 ms per output token when shared, 20,000 prompt tokens/s
            svc = r["input_tokens"] / 20000 + r["output_tokens"] * 0.025 * (1 + running / 40)
            events.append((r["arrival_s"] + svc, r["input_tokens"] + r["output_tokens"]))
        live = [(f, tok) for f, tok in events if f > now]
        running, used_tokens = len(live), sum(tok for _, tok in live)
        sidecar_gb = sidecar_steady_gb + sidecar_peak_growth_gb * min(1, running / 30)
        free_gb = total_gb - budget_gb - worker_ctx_gb - sidecar_gb
        need_gb = running * logit_row_bytes / GB
        row = {"t": f"{int((now % 86400) // 3600):02d}:{int((now % 3600) // 60):02d}:{int(now % 60):02d}", "num_requests_running": running,
               "kv_cache_usage_perc": round(min(1, used_tokens / pool_tokens), 3),
               "worker_pytorch_allocated_gb": round(weights_gb + activation_gb + pool_gb, 2),
               "sidecar_gb": round(sidecar_gb, 2), "gpu_free_gb": round(free_gb, 3), "sampler_needs_gb": round(need_gb, 3)}
        metrics.append(row)
        if need_gb > free_gb:
            crashed = row
            break
    return metrics, crashed


if __name__ == "__main__":
    metrics, crashed = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "incident2-metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
    c = crashed
    lines = [
        "# SIMULATED by simulate_oom_incident.py. Not from a real engine. Sizes computed from Qwen3-4B's config.json and the L4's 24 GB.",
        f"# weights {weights_gb:.2f} GB, KV pool {pool_gb:.2f} GB = {pool_tokens:,} tokens at {kv_per_token:,} B/token, "
        f"idle free memory outside the pool {free_idle:.2f} GB",
        f"{metrics[0]['t']} INFO  engine started: model=Qwen3-4B dtype=bfloat16 gpu_memory_utilization={UTIL} max_num_seqs=256",
        f"{metrics[0]['t']} INFO  KV cache: {pool_tokens:,} tokens ({pool_gb:.2f} GB)",
    ]
    if c:
        lines += [
            f"{c['t']} INFO  running={c['num_requests_running']} waiting=0 kv_cache_usage={c['kv_cache_usage_perc']*100:.1f}%",
            f"{c['t']} ERROR sampler: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate {c['sampler_needs_gb']*1000:.0f} MB "
            f"({c['num_requests_running']} rows x {V} x 4 B). GPU 0 has a total capacity of {total_gb:.0f} GB of which {c['gpu_free_gb']*1000:.0f} MB is free.",
            f"{c['t']} ERROR engine core died (exit code 1); {c['num_requests_running']} in-flight requests dropped; supervisor restarting",
        ]
    (OUT / "incident2-server.log").write_text("\n".join(lines) + "\n")
    print(f"weights {weights_gb:.2f} GB, pool {pool_gb:.2f} GB ({pool_tokens:,} tokens), idle free {free_idle:.2f} GB")
    print("crash at", c)
