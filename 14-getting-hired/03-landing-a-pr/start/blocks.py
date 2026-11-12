BLOCK = 16  # tokens per KV block


def blocks_needed(tokens: int) -> int:
    """How many KV blocks a prompt of this many tokens occupies."""
    raise NotImplementedError("TODO 14.3: copy blocks_buggy.py's idea, fix the bug (a partly filled last block still counts), see the lesson's 'Tests first, for a bug' section")
