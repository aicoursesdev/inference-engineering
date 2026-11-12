"""The project's checks on servinglab.kv.BlockAllocator, offline."""

import pytest

from servinglab.data import traces
from allocator import BlockAllocator, OutOfBlocks
from servinglab.kv import replay


def test_nothing_leaks_after_every_request_is_freed():
    a = BlockAllocator(40, 8)
    for i in range(5):
        p = list(range(50 + i))
        a.allocate(i, p)
        a.commit(i, p, len(p))
    for i in range(5):
        a.free(i)
    assert a.free_blocks == 40 and all(r == 0 for r in a.ref)


def test_a_shared_block_is_freed_only_when_its_last_user_leaves():
    a = BlockAllocator(16, 4)
    p = list(range(9))
    a.allocate("a", p); a.commit("a", p, 9)
    a.allocate("b", p)
    shared = a.seqs["a"].table[0]
    assert a.ref[shared] == 2
    a.free("a")
    assert a.ref[shared] == 1 and shared not in a.evictable
    a.free("b")
    assert shared in a.evictable


def test_lru_evicts_the_oldest_release_first():
    a = BlockAllocator(2, 4)
    a.allocate("x", [1, 2, 3, 4, 5]); a.commit("x", [1, 2, 3, 4, 5], 4); a.free("x")
    a.allocate("y", [9, 9, 9, 9, 8]); a.commit("y", [9, 9, 9, 9, 8], 4); a.free("y")
    a.allocate("z", [7, 7, 7, 7, 7, 7, 7, 7])        # needs both blocks
    assert a.evictions >= 1 and len(a.match([1, 2, 3, 4, 5])) == 0


def test_copy_on_write_leaves_the_other_sequence_untouched():
    a = BlockAllocator(8, 4)
    a.allocate("p", list(range(6)))
    a.fork("p", "q")
    before = list(a.seqs["p"].table)
    a.extend("q", 1)
    assert a.seqs["p"].table == before and a.seqs["q"].table[1] != before[1]


def test_allocation_fails_cleanly_when_the_pool_is_full():
    a = BlockAllocator(3, 4)
    a.allocate("a", list(range(12)))
    with pytest.raises(OutOfBlocks):
        a.allocate("b", list(range(4)))
    assert a.free_blocks == 0 and set(a.seqs) == {"a"}


def test_hit_rate_on_tidewater_is_the_system_prompt_share():
    rows = traces("tidewater")[:300]
    r = replay(rows, 3000, "shared")
    expected = sum(220 // 16 * 16 for _ in rows) / sum(x["input_tokens"] for x in rows)
    assert abs(r["hit_rate"] - expected) < 0.05
