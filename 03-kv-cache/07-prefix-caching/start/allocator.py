"""Lesson 03.7: block hashes and prefix matching.

Everything but the TODOs is finished. The finished class is BlockAllocator in
servinglab/kv.py; `make diff 03.7` shows what it has that this file lacks.
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
        raise NotImplementedError(
            "TODO 03.7: hash each full block with its parent's hash and "
            "return the cached block ids for the start of tokens, always "
            "leaving one token to compute, see the lesson's 'From block "
            "hashes to a tree' section")

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
        raise NotImplementedError(
            "TODO 03.7: register the hash of every block that n_done tokens"
            " fill completely, see the lesson's 'From block hashes to a "
            "tree' section")

    def extend(self, sid, n=1):
        """Make room for `n` more tokens. Returns the copies to make, a list
        of (from_block, to_block): a block shared with another sequence is
        copied before this one writes into it (copy-on-write)."""
        s, bs, copies = self.seqs[sid], self.block_size, []
        for _ in range(n):
            if s.n_tokens % bs == 0:                      # last block is full: a new one
                s.table.append(self._take())
            elif self.ref[s.table[-1]] > 1:               # shared and not full: copy it
                old = s.table[-1]
                new = self._take()
                self.fill[new] = self.fill[old]
                self.ref[old] -= 1                        # still held by the others
                s.table[-1] = new
                copies.append((old, new))
            self.fill[s.table[-1]] += 1
            s.n_tokens += 1
        return copies

    def fork(self, sid, child):
        """A second sequence that starts as a copy of `sid` (parallel samples,
        beam search): it shares every block until one of them writes."""
        s = self.seqs[sid]
        for b in s.table:
            self.ref[b] += 1
        self.seqs[child] = Seq(list(s.table), s.n_tokens, s.n_committed, list(s.chain))

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
        return sum(self.block_size - self.fill[b]
                   for b in range(self.num_blocks) if self.ref[b] > 0)
