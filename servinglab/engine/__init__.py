"""A Llama-architecture inference engine in plain NumPy (built in Module 01).

    from servinglab.engine import load, generate

    model = load("SmolLM2-135M-Instruct")      # downloads once, to models/
    ids = model.chat_ids([{"role": "user", "content": "Hi!"}])
    print(model.decode(generate(model, ids, max_new_tokens=40, temperature=0)))

    model.forward(ids)               # (T, vocab) logits, one row per token
    model.config, model.n_params     # shape and parameter count
    sample(logits, temperature=0.8, top_p=0.9)   # one token id

Every function is small and pure on purpose: later modules replace or wrap
single steps (the KV cache wraps `attention`, speculation calls `forward`).
Supports Llama-architecture checkpoints without rope scaling (SmolLM2).
"""

from .config import Config
from .generate import complete, generate
from .model import Model, attention, block, forward, load, rmsnorm, rope
from .sampling import apply_penalties, probabilities, sample, softmax

__all__ = ["Config", "Model", "load", "forward", "generate", "complete",
           "sample", "probabilities", "apply_penalties", "softmax",
           "attention", "block", "rmsnorm", "rope"]
