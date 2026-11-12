"""When does a verification pass stop being free? A back-of-envelope model.

One decode pass reads all the weights once (bytes / bandwidth) and does
2 FLOPs per weight per token in flight (FLOPs / peak). It takes the larger.
Verifying k guesses puts B * (k + 1) tokens in flight instead of B. KV-cache
reads and attention are left out, so real numbers are worse. Llama 3.1 8B in
bfloat16 on an H100 SXM, from the course's data files; an n-gram speculator
(draft cost 0) whose guesses are kept with probability alpha.

    uv run python ridge_model.py
"""

from servinglab.data import hardware, model_config

cfg, gpu = model_config("Llama-3.1-8B-Instruct"), hardware("H100-SXM")
h, L, ff = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["intermediate_size"]
kv = cfg["num_key_value_heads"] * h // cfg["num_attention_heads"]
n = L * (2 * h * h + 2 * h * kv + 3 * h * ff + 2 * h) + h + 2 * cfg["vocab_size"] * h
weights_bytes = 2 * n
mem_s = weights_bytes / (gpu["bandwidth_gbs"] * 1e9)
flop_s = 2 * n / (gpu["fp16_tflops"] * 1e12)            # per token in flight
print(f"{n / 1e9:.2f}B parameters, {weights_bytes / 1e9:.2f} GB; reading them "
      f"takes {mem_s * 1e3:.2f} ms; one token's maths {flop_s * 1e6:.1f} us; "
      f"the pass turns compute-bound at {mem_s / flop_s:.0f} tokens in flight")

alpha, k = 0.8, 4
tokens = (1 - alpha ** (k + 1)) / (1 - alpha)
print(f"alpha={alpha}, k={k}: {tokens:.2f} tokens per step")
print(f"{'batch':>6} {'plain ms':>9} {'spec ms':>8} {'speed-up':>9}")
for b in (1, 16, 64, 128, 200, 256, 512):
    plain = max(mem_s, flop_s * b)
    spec = max(mem_s, flop_s * b * (k + 1))
    print(f"{b:6d} {plain * 1e3:9.2f} {spec * 1e3:8.2f} {tokens * plain / spec:8.2f}x")
