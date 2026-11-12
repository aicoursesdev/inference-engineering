from allocator import BlockAllocator
from servinglab.data import traces
from servinglab.kv import replay


def test_waste_is_under_one_block_per_request():
    a = BlockAllocator(64, 16)
    a.allocate("x", list(range(100)))
    assert a.waste_slots() == 7 * 16 - 100          # 7 blocks of 16 hold 100 tokens


def test_extend_takes_a_new_block_only_when_the_last_is_full():
    a = BlockAllocator(8, 4)
    a.allocate("x", list(range(4)))
    assert len(a.seqs["x"].table) == 1
    a.extend("x", 1)
    assert len(a.seqs["x"].table) == 2 and a.waste_slots() == 3


def test_fork_shares_every_block():
    a = BlockAllocator(8, 4)
    a.allocate("p", list(range(6)))
    a.fork("p", "q")
    assert a.seqs["q"].table == a.seqs["p"].table
    assert all(a.ref[b] == 2 for b in a.seqs["p"].table)


def test_copy_on_write_leaves_the_other_sequence_untouched():
    a = BlockAllocator(8, 4)
    a.allocate("p", list(range(6)))
    a.fork("p", "q")
    before = list(a.seqs["p"].table)
    copies = a.extend("q", 1)
    assert a.seqs["p"].table == before and a.seqs["q"].table[1] != before[1]
    assert copies == [(before[1], a.seqs["q"].table[1])]
    assert a.ref[before[1]] == 1


def test_paging_wastes_far_less_than_reserving():
    rows = traces("tidewater")[:200]
    assert (replay(rows, 4000, "paged", max_len=2048)["mean_waste"]
            < 0.2 * replay(rows, 4000, "reserve", max_len=2048)["mean_waste"])
