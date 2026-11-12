"""Tensor parallelism on a laptop: split one MLP over 4 "GPUs", add the partial answers, count the bytes.

    uv run python tp_split.py

Workers are just slices of one NumPy array here. What matters is the arithmetic: cut the first
matrix by columns and the second by rows (Megatron-LM's split) and each worker's partial output
only needs adding together, once, at the end.
"""
import numpy as np
from servinglab.data import model_config

rng = np.random.default_rng(0)
H, I, WORKERS = 64, 256, 4                    # a small MLP: hidden 64, inner 256
x = rng.standard_normal((3, H))               # three tokens
w1, w2 = rng.standard_normal((H, I)), rng.standard_normal((I, H))
relu = lambda a: np.maximum(a, 0)             # any activation that acts on each number alone

whole = relu(x @ w1) @ w2
cols, rows = np.split(w1, WORKERS, axis=1), np.split(w2, WORKERS, axis=0)
partials = [relu(x @ c) @ r for c, r in zip(cols, rows)]   # each worker: its slice, no talking
print("largest difference, split vs whole:", np.abs(sum(partials) - whole).max())  # sum() is the all-reduce
print("one worker holds", cols[0].size + rows[0].size, "weights of", w1.size + w2.size)

print("\nWhat one all-reduce moves, Llama 3.1 70B, BF16 (hidden size from its config.json)")
cfg = model_config("Llama-3.1-70B-Instruct")
h, layers = cfg["hidden_size"], cfg["num_hidden_layers"]
for batch in (1, 32, 256):
    size = batch * h * 2                                     # one vector of h numbers per token, 2 bytes each
    for n in (2, 4, 8):
        per_gpu = 2 * (n - 1) / n * size                     # ring all-reduce: bytes each GPU sends
        t = per_gpu / 450e9 * 1e6                            # 450 GB/s: one direction of H100 NVLink (900 total)
        print(f"  batch {batch:3}: message {size/1e3:7.1f} kB, TP={n}: {per_gpu/1e3:7.1f} kB per GPU, "
              f">= {t:6.2f} us, x{2*layers} per step = {t*2*layers/1e3:5.2f} ms")

print("\nAgainst the step itself: each GPU reads its share of the weights from memory (Module 02's ceiling)")
from servinglab.data import hardware
from servinglab.roofline import shape
params = shape(cfg)["params"]
for n in (2, 4, 8):
    read_ms = params * 2 / n / (hardware("H100-SXM")["bandwidth_gbs"] * 1e9) * 1e3
    comm_ms = 2 * layers * (2 * (n - 1) / n * 32 * h * 2) / 450e9 * 1e3        # batch 32
    print(f"  TP={n}: weights read {read_ms:5.2f} ms, all-reduce floor {comm_ms:4.2f} ms ({comm_ms / read_ms * 100:3.0f}% on top)")
