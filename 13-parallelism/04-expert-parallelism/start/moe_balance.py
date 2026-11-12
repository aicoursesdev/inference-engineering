"""Expert load balance: how uneven routing makes one GPU the slowest, and what copies fix.

    uv run python moe_balance.py

DeepSeek-V3's config gives 256 routed experts and 8 chosen per token. We deal the experts
over GPUs and route 100,000 tokens with a skewed (Zipf-shaped) popularity. The skew is
SYNTHETIC: it stands in for the real, measured imbalance in the LMSYS and DeepSeek posts.
"""
import numpy as np
from servinglab.data import model_config

cfg = model_config("DeepSeek-V3")
E, K = cfg["n_routed_experts"], cfg["num_experts_per_tok"]
print(f"experts {E}, chosen per token {K}")
H, I, L = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["num_hidden_layers"] - cfg["first_k_dense_replace"]
one = 3 * H * I                                        # one expert: gate, up and down matrices
print(f"one expert {one / 1e6:.1f}M weights; all routed experts {E * one * L / 1e9:.0f}B over {L} layers; "
      f"a token uses {K}/{E} = {K / E * 100:.1f}% of them")
TOKENS, GPUS = 100_000, 32


def busiest(skew, extra=0):
    """Tokens on the busiest GPU divided by the average, with `extra` copies of the hottest experts."""
    rng = np.random.default_rng(0)                        # same draw for every call, so the rows compare fairly
    p = 1 / np.arange(1, E + 1) ** skew
    p = rng.permutation(p / p.sum())                       # which expert is hot is arbitrary
    load = np.zeros(E)
    for _ in range(TOKENS):                                # each token picks K distinct experts
        load[rng.choice(E, K, replace=False, p=p)] += 1
    copies = np.ones(E, int)
    for e in np.argsort(-load)[:extra]:
        copies[e] = 2                                      # a second copy of a hot expert splits its tokens
    slots = np.repeat(np.arange(E), copies)                # every copy is one slot
    share = np.repeat(load / copies, copies)
    order = np.argsort(-share)                             # greedy: biggest slot to the lightest GPU
    gpu = np.zeros(GPUS)
    for i in order:
        gpu[gpu.argmin()] += share[i]
    return gpu.max() / gpu.mean()


for skew in (0.0, 1.0, 1.5):
    print(f"skew {skew}: busiest GPU {busiest(skew):.2f}x the average; with 32 extra copies {busiest(skew, 32):.2f}x")
