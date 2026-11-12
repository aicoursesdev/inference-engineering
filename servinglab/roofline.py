"""Module 02: predict how fast a model runs on a chip from its config and the chip's datasheet.

    from servinglab.roofline import predict
    predict("Qwen3-0.6B", "H100-SXM")["decode"]["tokens_per_s"]
    predict("Qwen3-4B", "T4", batch=8, context=2000)["decode"]

A step takes as long as the slower of two jobs: moving its bytes from memory
and doing its arithmetic. That is all the model knows, so its answer is a
ceiling, never a promise: the lessons compare it with measurements.

Dense models only (Llama, Qwen, SmolLM). A mixture-of-experts config raises
ValueError until Module 17 teaches how experts change the count.
"""

from .data import hardware as _hardware
from .data import model_config as _model_config


def _chip(hardware):
    return _hardware(hardware) if isinstance(hardware, str) else dict(hardware)


def _config(model):
    return _model_config(model) if isinstance(model, str) else dict(model)


def shape(config):
    """The numbers a config tells us, in one dict (see lesson 01.5)."""
    if "num_local_experts" in config or "n_routed_experts" in config or "num_experts" in config:
        raise ValueError("mixture-of-experts models are not supported yet")
    h, layers, vocab = config["hidden_size"], config["num_hidden_layers"], config["vocab_size"]
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    head_dim = config.get("head_dim") or h // heads
    ffn = config["intermediate_size"]
    attn = h * heads * head_dim + 2 * h * kv_heads * head_dim + heads * head_dim * h
    mlp = 3 * h * ffn  # gate, up and down projections
    norms = 2 * h + (2 * head_dim if config.get("model_type") == "qwen3" else 0)
    per_layer = attn + mlp + norms
    embed = vocab * h
    # Decode multiplies by every matrix once per token, including the output head, which is the
    # embedding table itself when the model ties them.
    matmul = layers * (attn + mlp) + embed
    total = layers * per_layer + h + embed * (1 if config.get("tie_word_embeddings") else 2)
    return {
        "layers": layers, "heads": heads, "head_dim": head_dim,
        "params": total,               # every weight: what the file holds
        "matmul_params": matmul,       # the weights each token multiplies by
        "kv_bytes_per_token_f16": 2 * layers * kv_heads * head_dim * 2,
    }


def predict(model, hardware, batch=1, context=512, prompt=None,
            weight_bytes_per_param=2.0, kv_bytes_per_value=2.0, flops_key="fp16_tflops",
            overhead_seconds=0.0):
    """Predict one decode step and one prefill for `model` on `hardware`.

    model     a name in data/models/ or a config dict.
    hardware  a name in data/hardware.json or a dict with bandwidth_gbs and (for the compute
              side) a FLOPS entry named by `flops_key`; a chip with none (Apple) gets compute_ceiling None
              and is predicted by memory alone.
    batch     requests decoded together in one step.
    context   tokens each of them has already (their KV cache length).
    prompt    tokens in the prefill (defaults to `context`).
    overhead_seconds  time each step loses to launching work and other fixed costs; 0 gives the pure
              roofline, and lesson 02.4 shows how to fit one from a single measurement.
    weight_bytes_per_param  2.0 for float16, 1.0 for 8-bit, 0.5 for 4-bit (add the scales yourself:
              llama.cpp's Q8_0 is 34/32 = 1.0625).
    """
    s, chip, c = shape(_config(model)), _chip(hardware), context
    n = c if prompt is None else prompt
    bw = chip["bandwidth_gbs"] * 1e9
    peak = (chip.get(flops_key) or 0) * 1e12 or None
    wbytes = s["params"] * weight_bytes_per_param
    mm_bytes = s["matmul_params"] * weight_bytes_per_param  # what a step really reads
    kv_tok = s["kv_bytes_per_token_f16"] * kv_bytes_per_value / 2
    attn_dim = s["layers"] * s["heads"] * s["head_dim"]

    def step(flops, nbytes):
        t_mem = nbytes / bw
        t_cmp = flops / peak if peak else 0.0
        return {"flops": flops, "bytes": nbytes, "intensity": flops / nbytes,
                "seconds": max(t_mem, t_cmp) + overhead_seconds, "memory_seconds": t_mem, "compute_seconds": t_cmp,
                "bound": "compute" if t_cmp > t_mem else "memory"}

    decode = step(batch * (2 * s["matmul_params"] + 4 * attn_dim * c),
                  mm_bytes + batch * c * kv_tok)
    decode["tokens_per_s"] = batch / decode["seconds"]          # all users together
    decode["tokens_per_s_per_user"] = 1 / decode["seconds"]
    prefill = step(2 * s["matmul_params"] * n + 2 * attn_dim * n * n,   # causal: half of 4 * n * n
                   mm_bytes + n * kv_tok)
    prefill["tokens_per_s"] = n / prefill["seconds"]
    return {"model": s, "weight_bytes": wbytes, "decode": decode, "prefill": prefill,
            "ridge_flops_per_byte": peak / bw if peak else None}
