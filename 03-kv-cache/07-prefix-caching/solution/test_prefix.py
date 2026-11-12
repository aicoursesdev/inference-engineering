from allocator import BlockAllocator


def test_second_request_skips_the_shared_blocks():
    a = BlockAllocator(32, 16)
    system = list(range(64))
    a.allocate("a", system + [900])
    a.commit("a", system + [900], 65)
    assert a.allocate("b", system + [901]) == 64


def test_a_different_start_hits_nothing():
    a = BlockAllocator(32, 16)
    a.allocate("a", list(range(64)))
    a.commit("a", list(range(64)), 64)
    assert a.allocate("b", list(range(1000, 1064))) == 0


def test_nothing_matches_before_commit():
    a = BlockAllocator(32, 4)
    p = list(range(9))
    a.allocate("a", p)                       # allocated, keys not computed yet
    assert a.match(p) == []
    a.commit("a", p, 9)
    assert a.match(p) == a.seqs["a"].table[:2]


def test_one_token_is_always_left_to_compute():
    a = BlockAllocator(32, 4)
    p = list(range(8))                       # exactly two full blocks
    a.allocate("a", p)
    a.commit("a", p, 8)
    assert len(a.match(p)) == 1              # not 2: the last token needs logits


def test_a_block_is_named_by_its_whole_history():
    a = BlockAllocator(32, 4)
    a.allocate("a", [1, 2, 3, 4, 5, 6, 7, 8, 9])
    a.commit("a", [1, 2, 3, 4, 5, 6, 7, 8, 9], 9)
    # same second block, different first block: no hit at all
    assert a.match([9, 9, 9, 9, 5, 6, 7, 8, 9]) == []
