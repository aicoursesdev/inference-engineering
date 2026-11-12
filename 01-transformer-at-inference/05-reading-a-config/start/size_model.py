"""Size a model from its config.json alone: parameters and weight bytes.

    uv run python size_model.py
"""

from servinglab.data import model_config


def param_count(c):
    """Parameters from the six shape numbers (plus head size and a few flags)."""
    raise NotImplementedError(
        "TODO 01.5: add up attention (Q, K, V, O), the MLP (gate, up, down) and two norm weights "
        "per layer, times the layers, plus the embedding table, the output table unless tied and "
        "the final norm; Qwen3 adds 2 * head_dim per layer, see the lesson's 'The formula' section")


NAMES = ["SmolLM2-135M-Instruct", "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B",
         "Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct"]

if __name__ == "__main__":
    print(f"{'model':24s} {'parameters':>16s} {'bf16 GB':>8s} {'int4 GB':>8s}")
    for name in NAMES:
        n = param_count(model_config(name))
        print(f"{name:24s} {n:16,d} {n * 2 / 1e9:8.1f} {n * 0.5 / 1e9:8.1f}")
