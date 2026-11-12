BLOCK = 16  # tokens per KV block


def blocks_needed(tokens: int) -> int:
    """How many KV blocks a prompt of this many tokens occupies."""
    return -(-tokens // BLOCK)   # ceiling division: a partly filled block still counts
