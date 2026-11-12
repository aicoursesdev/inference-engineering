"""A model's shape, read from its config.json, and what that shape costs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    hidden: int          # width of the vector each token carries
    layers: int          # how many transformer blocks
    heads: int           # query heads per layer
    kv_heads: int        # key/value heads per layer (fewer than heads: GQA)
    head_dim: int        # numbers in each head
    intermediate: int    # width inside the MLP
    vocab: int           # how many tokens the model knows
    eps: float           # added inside RMSNorm so it never divides by zero
    rope_theta: float    # base of the rotary position frequencies
    tie_embeddings: bool  # does the output layer reuse the embedding table?
    eos: tuple           # token ids that end a reply

    @classmethod
    def from_json(cls, c):
        if c.get("model_type") != "llama":
            raise ValueError(f"only Llama-architecture models, got "
                             f"{c.get('model_type')!r}")
        if c.get("rope_scaling"):
            raise ValueError("rope_scaling is not supported by this engine")
        eos = c.get("eos_token_id", ())
        return cls(
            hidden=c["hidden_size"],
            layers=c["num_hidden_layers"],
            heads=c["num_attention_heads"],
            kv_heads=c.get("num_key_value_heads", c["num_attention_heads"]),
            head_dim=c.get("head_dim",
                           c["hidden_size"] // c["num_attention_heads"]),
            intermediate=c["intermediate_size"],
            vocab=c["vocab_size"],
            eps=c["rms_norm_eps"],
            rope_theta=c.get("rope_theta", 10000.0),
            tie_embeddings=c.get("tie_word_embeddings", False),
            eos=tuple(eos) if isinstance(eos, list) else (eos,),
        )

    def n_params(self):
        """Parameter count from the shape alone (lesson 01.5 derives this)."""
        d, kv, hd = self.hidden, self.kv_heads, self.head_dim
        attn = d * self.heads * hd + 2 * d * kv * hd + self.heads * hd * d
        mlp = 3 * d * self.intermediate
        norms = 2 * d
        total = self.layers * (attn + mlp + norms) + d  # + final norm
        total += self.vocab * d                          # embedding table
        if not self.tie_embeddings:
            total += self.vocab * d                      # separate output layer
        return total
