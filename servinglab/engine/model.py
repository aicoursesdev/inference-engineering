"""A Llama-architecture transformer in plain NumPy: the forward pass.

Shapes, with T = tokens in the input:

    ids            (T,)                token ids
    x              (T, hidden)         one vector per token, through the layers
    q              (T, heads, head_dim)
    k, v           (T, kv_heads, head_dim)
    logits         (T, vocab)          a score for every possible next token

Each function below is one step of the model. `forward` is the whole stack.
This engine recomputes every token on every call: the slow, simple baseline.
(Lesson 03.2 adds a cache so a call only computes the new tokens: `attention`
already accepts keys and values that are longer than the queries.)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .weights import read_safetensors

# The course repo's models/ folder, beside servinglab/.
MODELS = Path(__file__).resolve().parents[2] / "models"


@dataclass
class Model:
    config: Config
    weights: dict      # name -> float32 array, as in the safetensors file
    tokenizer: object  # a tokenizers.Tokenizer
    path: Path

    @property
    def n_params(self):
        return sum(w.size for w in self.weights.values())

    @property
    def n_bytes_bf16(self):
        """Bytes of the weights as the checkpoint stores them (2 per number)."""
        return 2 * self.n_params

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def chat_ids(self, messages):
        """Token ids for a conversation, ready for the model to answer.

        `messages` is [{"role": "user", "content": "..."}, ...]. This is the
        ChatML layout the SmolLM2 chat template produces (it adds its default
        system line when you give none); tests check it against the template.
        """
        text = ""
        if messages[0]["role"] != "system":
            text += ("<|im_start|>system\nYou are a helpful AI assistant "
                     "named SmolLM, trained by Hugging Face<|im_end|>\n")
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        return self.encode(text + "<|im_start|>assistant\n")

    def forward(self, ids, last_only=False):
        return forward(self, ids, last_only)


def load(name_or_path, models_dir=MODELS):
    """Load a model by folder name or path; download it first if missing.

    load("SmolLM2-135M-Instruct") looks in models/ of the course repo and, if
    the folder is not there, downloads HuggingFaceTB/SmolLM2-135M-Instruct
    (about 270 MB) with huggingface_hub.
    """
    from tokenizers import Tokenizer

    path = Path(name_or_path)
    if not path.is_dir():
        path = Path(models_dir) / name_or_path
    if not (path / "model.safetensors").is_file():
        from huggingface_hub import snapshot_download
        snapshot_download(f"HuggingFaceTB/{Path(name_or_path).name}",
                          local_dir=path,
                          allow_patterns=["config.json", "tokenizer.json",
                                          "model.safetensors"])
    config = Config.from_json(json.loads((path / "config.json").read_text()))
    weights = read_safetensors(path / "model.safetensors")
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    return Model(config, weights, tokenizer, path)


# ---- the steps ----------------------------------------------------------

def rmsnorm(x, weight, eps):
    """Scale each token's vector to unit root-mean-square, then by weight."""
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope(x, positions, theta):
    """Rotate pairs of numbers in each head by an angle set by position.

    x is (T, heads, head_dim). The first half of each head is paired with
    the second half (the layout Llama checkpoints use).
    """
    half = x.shape[-1] // 2
    freqs = theta ** (-np.arange(half) / half)          # (half,)
    angles = positions[:, None] * freqs[None, :]        # (T, half)
    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    a, b = x[..., :half], x[..., half:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)


def attention(q, k, v):
    """Causal attention. q is (T, heads, d); k and v are (S, kv_heads, d)
    with S >= T: the queries are the last T of the S positions.

    Several query heads share each key/value head (grouped-query attention).
    """
    T, heads, d = q.shape
    S, kv_heads, _ = k.shape
    group = heads // kv_heads
    k = np.repeat(k, group, axis=1)                     # (S, heads, d)
    v = np.repeat(v, group, axis=1)
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(d)   # (heads, T, S)
    # query t sits at position S - T + t and may see keys 0 .. that position
    visible = np.arange(S)[None, :] <= (S - T + np.arange(T))[:, None]
    scores = np.where(visible[None], scores, -np.inf)
    scores -= scores.max(-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(-1, keepdims=True)
    return np.einsum("hts,shd->thd", w, v).reshape(T, heads * d)


def block(model, i, x, positions):
    """One transformer block: attention then MLP, each added back (residual)."""
    c, w = model.config, model.weights
    p = f"model.layers.{i}."
    h = rmsnorm(x, w[p + "input_layernorm.weight"], c.eps)
    T = len(x)
    q = (h @ w[p + "self_attn.q_proj.weight"].T).reshape(T, c.heads, c.head_dim)
    k = (h @ w[p + "self_attn.k_proj.weight"].T).reshape(T, c.kv_heads, c.head_dim)
    v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(T, c.kv_heads, c.head_dim)
    q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
    x = x + attention(q, k, v) @ w[p + "self_attn.o_proj.weight"].T
    h = rmsnorm(x, w[p + "post_attention_layernorm.weight"], c.eps)
    gate = silu(h @ w[p + "mlp.gate_proj.weight"].T)
    up = h @ w[p + "mlp.up_proj.weight"].T
    return x + (gate * up) @ w[p + "mlp.down_proj.weight"].T


def forward(model, ids, last_only=False):
    """Logits for the next token after each position of `ids`.

    Returns (T, vocab), or (1, vocab) with last_only=True (generation only
    needs the last row, and the output layer is the biggest single matmul).
    """
    c, w = model.config, model.weights
    ids = np.asarray(ids)
    x = w["model.embed_tokens.weight"][ids]             # (T, hidden)
    positions = np.arange(len(ids))
    for i in range(c.layers):
        x = block(model, i, x, positions)
    if last_only:
        x = x[-1:]
    x = rmsnorm(x, w["model.norm.weight"], c.eps)
    head = w["model.embed_tokens.weight" if c.tie_embeddings
             else "lm_head.weight"]
    return x @ head.T
