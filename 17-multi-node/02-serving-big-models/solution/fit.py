"""How many GPUs does a model need? Weights first, then KV cache."""
import json, math
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data" / "models"
GB = 1e9
# GPU memory in GB (NVIDIA product pages for H100 and H200, Nov 2026)
GPUS = {"H100 80GB": 80, "H200 141GB": 141}
USABLE = 0.90  # leave 10% for activations, CUDA context, fragmentation


def llama_params(c):
    h, ff, L = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    d = h // c["num_attention_heads"]
    kv = c["num_key_value_heads"] * d
    attn = h * h + 2 * h * kv + h * h          # q, k, v, o
    mlp = 3 * h * ff                            # gate, up, down
    return L * (attn + mlp) + 2 * c["vocab_size"] * h   # + embed, lm_head


def kv_bytes_per_token(c, bytes_per=2):
    d = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    return 2 * c["num_hidden_layers"] * c["num_key_value_heads"] * d * bytes_per


def gpus_needed(weights_gb, kv_gb, gpu_gb):
    return math.ceil((weights_gb + kv_gb) / (USABLE * gpu_gb))


def next_valid(n, kv_heads):
    """TP degree must divide the KV heads; pick the next size that does,
    or a multiple of it (replicate KV heads) once it passes the head count."""
    while kv_heads % n and n % kv_heads:
        n += 1
    return n


if __name__ == "__main__":
    cfg = json.load(open(DATA / "Llama-3.1-70B-Instruct.json"))
    n70 = llama_params(cfg)
    print(f"Llama 3.1 70B from its config: {n70/1e9:.2f}B parameters")
    per_tok = kv_bytes_per_token(cfg)
    print(f"KV cache per token (BF16): {per_tok:,} bytes")
    ctx, batch = 8192, 32
    kv_gb = per_tok * ctx * batch / GB
    print(f"KV for {batch} requests of {ctx} tokens: {kv_gb:.1f} GB\n")
    rows = [("70B BF16", n70 * 2 / GB, kv_gb, 8),
            ("70B FP8", n70 * 1 / GB, per_tok / 2 * ctx * batch / GB, 8),
            ("400B BF16 (weights only)", 400e9 * 2 / GB, 0, 8),
            ("400B FP8 (weights only)", 400e9 * 1 / GB, 0, 8)]
    print(f"{'case':26}{'weights GB':>11}{'KV GB':>8}", end="")
    for g in GPUS:
        print(f"{g:>14}", end="")
    print()
    for name, w, k, kvh in rows:
        print(f"{name:26}{w:11.0f}{k:8.0f}", end="")
        for g, mem in GPUS.items():
            n = next_valid(gpus_needed(w, k, mem), kvh)
            print(f"{n:>9} GPUs ", end="")
        print()
