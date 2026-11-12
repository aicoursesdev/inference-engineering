"""The KV cache and the block allocator (Module 03). Offline: a tiny model, no network."""

import json
from pathlib import Path

import numpy as np
import pytest

from servinglab.data import traces
from servinglab.engine import Config, Model, generate
from servinglab.engine.weights import read_safetensors
from servinglab.kv import (BlockAllocator, KVCache, OutOfBlocks, PagedCache,
                           forward_cached, generate_cached, kv_bytes_per_token,
                           replay, step_forward, trace_tokens)

DATA = Path(__file__).parent / "data" / "tiny-llama"


def tiny():
    config = Config.from_json(json.loads((DATA / "config.json").read_text()))
    return Model(config, read_safetensors(DATA / "model.safetensors"), None, DATA)


IDS = [5, 17, 3, 99, 42, 7, 7, 61, 2, 90, 14, 33, 8, 21, 50, 4, 77, 12, 6, 88]


# ---- the cache ----

def test_cached_prefill_matches_the_full_forward():
    m = tiny()
    cache = KVCache(m.config, 64)
    assert np.allclose(forward_cached(m, IDS, cache), m.forward(IDS)[-1], atol=1e-4)


def test_decode_one_token_at_a_time_matches_the_full_forward():
    m, cache = tiny(), KVCache(tiny().config, 64)
    forward_cached(m, IDS[:12], cache)
    for i in range(12, len(IDS)):
        row = forward_cached(m, [IDS[i]], cache)
        assert np.allclose(row, m.forward(IDS[:i + 1])[-1], atol=1e-4)
    assert cache.length == len(IDS)


def test_cached_generation_is_the_same_tokens():
    m = tiny()
    assert (list(generate_cached(m, IDS, 20, temperature=0))
            == list(generate(m, IDS, 20, temperature=0)))


def test_a_batch_step_gives_each_sequence_its_own_answer():
    m = tiny()
    a, b = KVCache(m.config, 64), KVCache(m.config, 64)
    rows = step_forward(m, [(a, IDS[:9]), (b, IDS[5:])])
    assert np.allclose(rows[0], m.forward(IDS[:9])[-1], atol=1e-4)
    assert np.allclose(rows[1], m.forward(IDS[5:])[-1], atol=1e-4)


def test_cache_refuses_to_overflow():
    with pytest.raises(ValueError):
        forward_cached(tiny(), IDS, KVCache(tiny().config, 5))


def test_kv_bytes_per_token_from_the_config():
    c = tiny().config
    assert kv_bytes_per_token(c) == 2 * c.layers * c.kv_heads * c.head_dim * 2


# ---- the allocator ----

def test_blocks_are_allocated_by_ceiling_and_freed():
    a = BlockAllocator(8, 4)
    assert a.allocate("x", list(range(10))) == 0
    assert len(a.seqs["x"].table) == 3 and a.free_blocks == 5
    a.free("x")
    assert a.free_blocks == 8


def test_out_of_blocks_changes_nothing():
    a = BlockAllocator(2, 4)
    with pytest.raises(OutOfBlocks):
        a.allocate("x", list(range(9)))
    assert a.free_blocks == 2 and "x" not in a.seqs


def test_a_second_request_reuses_a_committed_prefix():
    a = BlockAllocator(16, 4)
    system = list(range(100, 112))                 # three full blocks
    p1, p2 = system + [1, 2, 3], system + [7, 8, 9, 10, 11]
    a.allocate("a", p1)
    assert a.allocate("b", p2) == 0                # nothing committed yet: no hit
    a.free("b")
    a.commit("a", p1, len(p1))
    assert a.allocate("c", p2) == 12               # the three shared blocks
    assert a.seqs["c"].table[:3] == a.seqs["a"].table[:3]
    assert all(a.ref[b] == 2 for b in a.seqs["c"].table[:3])
    a.free("a")
    assert all(a.ref[b] == 1 for b in a.seqs["c"].table[:3])


def test_a_fully_cached_prompt_still_recomputes_its_last_token():
    a = BlockAllocator(8, 4)
    p = list(range(8))
    a.allocate("a", p)
    a.commit("a", p, 8)
    assert a.allocate("b", p) == 4                 # not 8: the logits need one token


def test_freed_prefix_stays_cached_until_evicted():
    a = BlockAllocator(4, 4)
    p = list(range(9))
    a.allocate("a", p)
    a.commit("a", p, 9)
    a.free("a")
    assert a.free_blocks == 4 and len(a.cached) == 2
    assert a.allocate("b", p) == 8                 # revived from the free pool
    a.free("b")
    a.allocate("c", list(range(500, 516)))         # needs all four: evicts
    assert len(a.cached) == 0 and a.evictions == 2


def test_eviction_takes_the_end_of_a_prefix_first():
    a = BlockAllocator(3, 4)
    p = list(range(9))                             # blocks r, s, and a partial third
    a.allocate("a", p)
    a.commit("a", p, 9)
    a.free("a")
    a.allocate("b", list(range(500, 508)))         # needs two: the clean one, then s
    assert a.evictions == 1
    assert len(a.match(p)) == 1                    # and a new prompt can still hit it
    assert len(a.cached) == 1                      # the root block r survives


def test_copy_on_write_when_a_forked_sequence_writes():
    a = BlockAllocator(8, 4)
    a.allocate("a", list(range(6)))                # blocks: one full, one half
    a.fork("a", "b")
    shared = a.seqs["a"].table[1]
    assert a.ref[shared] == 2
    copies = a.extend("b", 1)
    assert copies == [(shared, a.seqs["b"].table[1])]
    assert a.ref[shared] == 1 and a.seqs["a"].table[1] == shared
    assert a.seqs["a"].table[0] == a.seqs["b"].table[0]


def test_extend_opens_a_new_block_on_a_boundary():
    a = BlockAllocator(8, 4)
    a.allocate("a", list(range(4)))
    a.extend("a", 1)
    assert len(a.seqs["a"].table) == 2 and a.fill[a.seqs["a"].table[1]] == 1


# ---- real arrays in blocks ----

def test_paged_cache_matches_the_contiguous_one_and_shares_a_prefix():
    m = tiny()
    cache = PagedCache(m.config, num_blocks=16, block_size=4)
    a, skipped = cache.start("a", IDS)
    assert skipped == 0
    la = step_forward(m, [(a, IDS)])[0]
    cache.commit("a", IDS)
    assert np.allclose(la, m.forward(IDS)[-1], atol=1e-4)
    ids2 = IDS[:16] + [1, 2, 3]
    b, skipped = cache.start("b", ids2)
    assert skipped == 16
    lb = step_forward(m, [(b, ids2[skipped:])])[0]
    assert np.allclose(lb, m.forward(ids2)[-1], atol=1e-4)


def test_paged_decode_after_a_copy_on_write_stays_correct():
    m = tiny()
    cache = PagedCache(m.config, 16, 4)
    a, _ = cache.start("a", IDS[:10])
    step_forward(m, [(a, IDS[:10])])
    cache.alloc.fork("a", "b")
    cache.computed["b"] = cache.computed["a"]
    b = type(a)(cache, "b")
    cache.grow("a", 1)
    cache.grow("b", 1)
    la = step_forward(m, [(a, [IDS[10]])])[0]
    lb = step_forward(m, [(b, [IDS[11]])])[0]
    assert np.allclose(la, m.forward(IDS[:10] + [IDS[10]])[-1], atol=1e-4)
    assert np.allclose(lb, m.forward(IDS[:10] + [IDS[11]])[-1], atol=1e-4)


# ---- the trace report ----

def test_trace_tokens_share_exactly_the_declared_prefix():
    rows = traces("tidewater")[:2]
    a, b = trace_tokens(rows[0]), trace_tokens(rows[1])
    assert len(a) == rows[0]["input_tokens"]
    p = rows[0]["prefix_tokens"]
    assert a[:p] == b[:p] and a[p] != b[p]


def test_replay_ranks_the_three_policies():
    rows = traces("tidewater")[:300]
    out = {m: replay(rows, 2000, m) for m in ("reserve", "paged", "shared")}
    assert out["reserve"]["mean_waste"] > out["paged"]["mean_waste"]
    assert out["shared"]["hit_rate"] > 0.3 and out["paged"]["hit_rate"] == 0
    assert out["shared"]["peak_slots"] <= out["paged"]["peak_slots"]
