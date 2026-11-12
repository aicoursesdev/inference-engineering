BLOCK = 16  # tokens per KV block


def blocks_needed(tokens: int) -> int:
    """How many KV blocks a prompt of this many tokens occupies."""
    return tokens // BLOCK       # the bug: drops a partly filled last block
