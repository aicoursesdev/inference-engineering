"""Prove the repo works: read the data files and print what is in them.

    uv run python first_look.py
"""

import statistics

from servinglab.data import hardware, model_config, traces

for name in ("T4", "H100-SXM", "M5-Max"):
    h = hardware(name)
    print(f"{name:9s} {h['memory_gb']:>4} GB memory, "
          f"{h['bandwidth_gbs']:>5} GB/s bandwidth")

cfg = model_config("SmolLM2-135M-Instruct")
print("SmolLM2-135M:", cfg["num_hidden_layers"], "layers, vector size",
      cfg["hidden_size"])

for name in ("tidewater", "kodekart", "nyaya", "batch"):
    t = traces(name)
    median = statistics.median(r["input_tokens"] for r in t)
    print(f"{name:10s} {len(t):5d} requests, "
          f"median prompt {median:7.1f} tokens")
