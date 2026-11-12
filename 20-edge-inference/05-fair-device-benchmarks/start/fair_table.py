"""Put engines on one footing: speed per byte of file, then the rest.

Different engines may read different files, so tokens/s alone is not
comparable. Tokens/s x file size = bytes read per second.
"""
ROWS = [  # engine, tokens/s (results_m5max.txt), file bytes
    ("llama-server", 357, 522_640_096),    # Ollama's own GGUF, Q4_K_M
    ("Ollama",       280, 522_640_096),    # the same file
    ("mlx_lm.server", 414, 335_450_584),   # MLX 4-bit, groups of 64
]

if __name__ == "__main__":
    print(f"{'engine':14s} {'tok/s':>6s} {'file MB':>8s} {'GB/s read':>10s}")
    for name, tps, size in ROWS:
        print(f"{name:14s} {tps:6d} {size / 1e6:8.0f} {tps * size / 1e9:10.0f}")
