"""Size a model from its config.json alone: parameters and weight bytes.

    uv run python size_model.py
"""

from servinglab.data import model_config


def param_count(c):
    """Parameters from the six shape numbers (plus head size and a few flags)."""
    d = c["hidden_size"]
    heads, kv = c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim") or d // heads
    attn = d * heads * hd + 2 * d * kv * hd + heads * hd * d  # Q, K and V, O
    if c["model_type"] == "qwen3":
        attn += 2 * hd                    # Qwen3 normalises each Q and K head
    mlp = 3 * d * c["intermediate_size"]  # gate, up, down
    layer = attn + mlp + 2 * d            # + two norm weights
    embed = c["vocab_size"] * d
    out = 0 if c["tie_word_embeddings"] else embed
    return c["num_hidden_layers"] * layer + embed + out + d


NAMES = ["SmolLM2-135M-Instruct", "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B",
         "Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct"]

if __name__ == "__main__":
    print(f"{'model':24s} {'parameters':>16s} {'bf16 GB':>8s} {'int4 GB':>8s}")
    for name in NAMES:
        n = param_count(model_config(name))
        print(f"{name:24s} {n:16,d} {n * 2 / 1e9:8.1f} {n * 0.5 / 1e9:8.1f}")
