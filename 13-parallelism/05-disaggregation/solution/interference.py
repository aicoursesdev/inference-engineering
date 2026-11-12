"""Why a long prompt freezes everyone else's reply, and what moving its KV costs.

    uv run python interference.py

Roofline ceilings from Module 02's predictor (no efficiency factors), Llama 3.1 8B on an H100.
"""
from servinglab.roofline import predict

MODEL, CHIP = "Llama-3.1-8B-Instruct", "H100-SXM"
chat = predict(MODEL, CHIP, batch=32, context=600)["decode"]      # 32 Tidewater chats, 600 tokens each
contract = predict(MODEL, CHIP, prompt=35_000)["prefill"]         # one Nyaya contract
print(f"one decode step for 32 chats:  {chat['seconds'] * 1e3:6.1f} ms  ({chat['bound']}-bound)")
print(f"prefill of a 35,000-token contract: {contract['seconds'] * 1e3:6.0f} ms  ({contract['bound']}-bound)")
print(f"the chats wait {contract['seconds'] / chat['seconds']:.0f} decode steps for one prompt, if it runs in one piece")

kv = predict(MODEL, CHIP, context=35_000)["model"]["kv_bytes_per_token_f16"] * 35_000
for name, bw in [("400 Gb/s network card (50 GB/s)", 50e9), ("NVLink, one direction (450 GB/s)", 450e9)]:
    print(f"moving its {kv / 1e9:.1f} GB of KV over {name}: {kv / bw * 1e3:6.1f} ms "
          f"= {kv / bw / contract['seconds'] * 100:.0f}% of the prefill")
