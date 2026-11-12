"""Mixture-of-experts in NumPy: route, group by expert, multiply, combine.

`align_block_size` follows the idea of vLLM's function of the same name
(vllm/model_executor/layers/fused_moe/moe_align_block_size.py): sort the
(token, expert) pairs by expert and pad each expert's group to a multiple of
the block size, so every GPU block works on one expert's weights.
"""
import numpy as np


def route(logits, k):
    """Top-k experts per token and their weights (softmax over the k)."""
    ids = np.argsort(-logits, axis=1)[:, :k]
    top = np.take_along_axis(logits, ids, axis=1)
    w = np.exp(top - top.max(axis=1, keepdims=True))
    return ids, w / w.sum(axis=1, keepdims=True)


def align_block_size(topk_ids, block, num_experts):
    """Returns (sorted_ids, expert_of_block, total).
    A route is numbered token * k + slot; the pad value is topk_ids.size."""
    raise NotImplementedError("TODO 16.6: sort routes by expert, pad each expert's group to a multiple of block with the pad value topk_ids.size, see the lesson's 'Grouping the routes by expert' section")


def moe_naive(x, W, topk_ids, topk_w):
    """One tiny matmul per (token, expert) route: the slow reference."""
    out = np.zeros((x.shape[0], W.shape[2]))
    for t in range(x.shape[0]):
        for slot in range(topk_ids.shape[1]):
            out[t] += topk_w[t, slot] * (x[t] @ W[topk_ids[t, slot]])
    return out


def moe_grouped(x, W, topk_ids, topk_w, block=4):
    """One matmul per block of routes that share an expert."""
    raise NotImplementedError("TODO 16.6: one real matmul per block of routes that share an expert, then np.add.at into each token's row, scaled by its router weight, see the lesson's 'Grouping the routes by expert' section")


def warp_decode(a, W_down, ids, w, lanes=32):
    """Cursor-style decode for ONE token: each output value is owned by one
    warp. Lane l sums inputs l, l+32, ... across the token's k experts into a
    float32 running total; the lanes' partials are then added. a: (k, I),
    W_down: (E, H, I), ids and w: (k,)."""
    raise NotImplementedError("TODO 16.6: one warp per output value h: lane l sums inputs l, l+32, ... across the k experts in float32, then add the lanes, see the lesson's 'Warp decode: flip the axis' section")


def expected_experts_touched(experts, k, batch):
    """Distinct experts a decode batch reads, if every token picked its k
    experts uniformly and independently (real routers are less even)."""
    return experts * (1 - (1 - k / experts) ** batch)


if __name__ == "__main__":
    # The shape of vLLM's docstring example: 4 tokens, 3 experts each, 4 experts.
    ids = np.array([[1, 2, 3], [0, 1, 2], [0, 2, 3], [0, 1, 3]])
    s, e, total = align_block_size(ids, block=4, num_experts=4)
    print("sorted_ids      :", s.tolist(), "(12 = padding)")
    print("expert_of_block :", e.tolist(), " total:", total)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 16))
    W = rng.normal(size=(8, 16, 12))
    ids, w = route(rng.normal(size=(20, 8)), 2)
    print("grouped equals naive:",
          np.allclose(moe_naive(x, W, ids, w), moe_grouped(x, W, ids, w)))

    a = rng.normal(size=(4, 64)).astype(np.float32)
    Wd = rng.normal(size=(8, 10, 64)).astype(np.float32)
    ids1, w1 = np.array([5, 1, 7, 2]), np.array([.4, .3, .2, .1], np.float32)
    ref = sum(w1[s_] * (Wd[ids1[s_]] @ a[s_]) for s_ in range(4))
    print("warp decode equals reference:", np.allclose(warp_decode(a, Wd, ids1, w1), ref, atol=1e-4))

    for name, E, k in (("gpt-oss-20b", 32, 4), ("DeepSeek-V3", 256, 8)):
        row = ", ".join(f"batch {b}: {expected_experts_touched(E, k, b):.1f}"
                        for b in (1, 4, 16, 64))
        print(f"{name} ({E} experts, top {k}) experts read per layer: {row}")
