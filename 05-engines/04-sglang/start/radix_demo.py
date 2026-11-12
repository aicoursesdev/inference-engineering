"""Count how many prompt tokens a prefix tree lets you skip (a sketch).

A real radix tree (SGLang, mini-sglang) merges runs of tokens into one
edge; this plain tree keeps one token per node, which is enough to count.
"""
SYSTEM = list(range(3000))          # Kodekart's 3,000-token system prompt


def reused(tree, prompt):
    """Walk the tree along the prompt; return tokens found, then add it."""
    node, hit = tree, 0
    for tok in prompt:
        if tok in node:
            hit += 1
        node = node.setdefault(tok, {})
    return hit


if __name__ == "__main__":
    tree, step = {}, SYSTEM + [9001, 9002]            # step 1
    for n in range(1, 4):
        prompt = step
        hit = reused(tree, prompt)
        print(f"step {n}: prompt {len(prompt)} tokens, "
              f"reused {hit}, computed {len(prompt) - hit}")
        step = prompt + [7000 + n] * 400               # reply + tool result
    other = SYSTEM + [4242] * 50                       # a different task
    hit = reused(tree, other)
    print(f"new task: prompt {len(other)}, reused {hit}, "
          f"computed {len(other) - hit}")
