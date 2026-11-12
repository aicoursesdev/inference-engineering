"""Lesson 03.6: blocks, a block table and copy-on-write.

Everything but the TODOs is finished. The finished class is BlockAllocator in
servinglab/kv.py; `make diff 03.6` shows what it has that this file lacks.
"""

from collections import OrderedDict
import math

from servinglab.kv import OutOfBlocks, Seq   # noqa: F401


class BlockAllocator:
    """Fixed-size blocks, reference counts, copy-on-write and a prefix cache.

    A block is `block_size` token slots. A sequence owns a *block table*: the
    ids of its blocks in order. Two sequences can list the same block (its
    reference count says how many do). A full block that has been filled is
    named by a hash of its tokens and of the block before it, so the same
    prompt start always lands on the same blocks.

    When the pool holds only cached-but-unused blocks, the one released
    longest ago is evicted (LRU). A sequence releases its blocks last-first,
    so the end of a prefix goes before its start: the order a radix tree
    gives by evicting leaves first, with no tree to keep.
    """

    def __init__(self, num_blocks, block_size=16):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.ref = [0] * num_blocks
        self.fill = [0] * num_blocks          # slots in use inside each block
        self.hash = [None] * num_blocks       # set when a full block is committed
        self.clean = OrderedDict((b, None) for b in range(num_blocks))  # unused, no content
        self.evictable = OrderedDict()        # unused but cached, oldest release first
        self.cached = {}                      # hash -> block id
        self.seqs = {}
        self.hit_tokens = self.query_tokens = 0
        self.evictions = 0

    # -- the pool --
    @property
    def free_blocks(self):
        """Blocks a new request could take: unused ones, cached or not."""
        return len(self.clean) + len(self.evictable)

    @property
    def used_blocks(self):
        return self.num_blocks - self.free_blocks

    def _take(self):
        if self.clean:
            b, _ = self.clean.popitem(last=False)
        else:
            if not self.evictable:
                raise OutOfBlocks("no free block")
            b, _ = self.evictable.popitem(last=False)
            self._forget(b)
            self.evictions += 1
        self.ref[b], self.fill[b] = 1, 0
        return b

    def _forget(self, b):
        h = self.hash[b]
        if h is not None:
            del self.cached[h]
            self.hash[b] = None

    def _release(self, b):
        self.ref[b] -= 1
        if self.ref[b] == 0:
            if self.hash[b] is None:
                self.clean[b] = None
            else:
                self.evictable[b] = None          # newest release goes last

    def _chain(self, tokens, upto_blocks, first_parent=None):
        """Hashes of the first `upto_blocks` full blocks of `tokens`."""
        out, parent, bs = [], first_parent, self.block_size
        for j in range(upto_blocks):
            parent = hash((parent, tuple(tokens[j * bs:(j + 1) * bs])))
            out.append(parent)
        return out

    # -- a sequence's life --
    def match(self, tokens):
        """Block ids already cached for the start of `tokens`. At least one
        token is always left to compute, because a request needs the logits
        of its last token."""
        bs = self.block_size
        hashes = self._chain(tokens, (len(tokens) - 1) // bs)
        hits = []
        for h in hashes:
            b = self.cached.get(h)
            if b is None:
                break
            hits.append(b)
        return hits

    def can_allocate(self, tokens, share=True):
        hits = self.match(tokens) if share else []
        need = math.ceil(len(tokens) / self.block_size) - len(hits)
        revived = sum(1 for b in hits if self.ref[b] == 0)
        return need <= self.free_blocks - revived

    def allocate(self, sid, tokens, share=True):
        """Give sequence `sid` blocks for the prompt `tokens`. Cached blocks
        of a shared start are reused, not recomputed. Returns the number of
        tokens already in the cache (a multiple of the block size)."""
        if sid in self.seqs:
            raise ValueError(f"sequence {sid!r} already exists")
        hits = self.match(tokens) if share else []
        need = math.ceil(len(tokens) / self.block_size) - len(hits)
        revived = sum(1 for b in hits if self.ref[b] == 0)
        if need > self.free_blocks - revived:
            raise OutOfBlocks(f"need {need} new blocks, {self.free_blocks - revived} free")
        table = []
        for b in hits:
            if self.ref[b] == 0:
                del self.evictable[b]
            self.ref[b] += 1
            table.append(b)
        for j in range(len(hits), math.ceil(len(tokens) / self.block_size)):
            b = self._take()
            self.fill[b] = min(self.block_size, len(tokens) - j * self.block_size)
            table.append(b)
        cached_tokens = len(hits) * self.block_size
        self.seqs[sid] = Seq(table, len(tokens), cached_tokens,
                             [self.hash[b] for b in hits])
        self.hit_tokens += cached_tokens
        self.query_tokens += len(tokens)
        return cached_tokens

    def commit(self, sid, tokens, n_done):
        """Say that the first `n_done` tokens of `sid` are now computed. Every
        block they fill completely becomes a cached prefix others can reuse."""
        s, bs = self.seqs[sid], self.block_size
        full = n_done // bs
        for j in range(len(s.chain), full):
            b = s.table[j]
            parent = s.chain[-1] if s.chain else None
            h = hash((parent, tuple(tokens[j * bs:(j + 1) * bs])))
            s.chain.append(h)
            if self.hash[b] is None and h not in self.cached:
                self.hash[b] = h
                self.cached[h] = b
        s.n_committed = max(s.n_committed, full * bs)

    def extend(self, sid, n=1):
        """Make room for `n` more tokens. Returns the copies to make, a list
        of (from_block, to_block): a block shared with another sequence is
        copied before this one writes into it (copy-on-write)."""
        raise NotImplementedError(
            "TODO 03.6: grow by n tokens: a new block when the last is "
            "full, and when the last block is shared copy it first and "
            "return (from_block, to_block), see the lesson's 'Sharing and "
            "copy-on-write' section")

    def fork(self, sid, child):
        """A second sequence that starts as a copy of `sid` (parallel samples,
        beam search): it shares every block until one of them writes."""
        raise NotImplementedError(
            "TODO 03.6: give the child a copy of the parent's table and "
            "count every shared block once more, see the lesson's 'Sharing "
            "and copy-on-write' section")

    def free(self, sid):
        """Release a sequence. Blocks go back last-first, so a cached prefix is
        evicted from its end, and blocks no one else holds are reusable."""
        for b in reversed(self.seqs.pop(sid).table):
            self._release(b)

    # -- numbers for reports --
    @property
    def hit_rate(self):
        return self.hit_tokens / self.query_tokens if self.query_tokens else 0.0

    def waste_slots(self):
        """Token slots held by blocks in use but holding nothing yet."""
        raise NotImplementedError(
            "TODO 03.6: return the slots held by blocks in use that hold "
            "nothing yet, see the lesson's 'Blocks and a block table' "
            "section")
