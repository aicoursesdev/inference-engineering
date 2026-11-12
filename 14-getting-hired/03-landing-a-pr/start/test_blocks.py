import importlib
import os

# The module under test: "blocks" (fixed) or "blocks_buggy". Lets one test run both ways.
blocks = importlib.import_module(os.environ.get("BLOCKS_MODULE", "blocks"))


def test_partly_filled_block_still_counts():
    # 17 tokens fill one block and start a second: two blocks, not one.
    assert blocks.blocks_needed(17) == 2


def test_exact_multiple_and_empty():
    assert blocks.blocks_needed(32) == 2
    assert blocks.blocks_needed(0) == 0
