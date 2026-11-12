"""Project 03.9: your block allocator with prefix sharing.

Everything but the TODOs is finished. The finished class is BlockAllocator in
servinglab/kv.py; `make diff 03.9` shows what it has that this file lacks.
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
        raise NotImplementedError(
            "TODO 03.9: step 5: hand out an unused block, evicting the "
            "oldest cached one when none is clean, and raise OutOfBlocks "
            "when there is none, see the lesson's 'Steps' section")

    def _forget(self, b):
        h = self.hash[b]
        if h is not None:
            del self.cached[h]
            self.hash[b] = None

    def _release(self, b):
        raise NotImplementedError(
            "TODO 03.9: step 1: lower a block's count and, at zero, put it "
            "back as clean or evictable, see the lesson's 'Steps' section")

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
            "TODO 03.9: step 3: return the cached block ids for the start "
            "of tokens, leaving one token to compute, see the lesson's "
            "'Steps' section")

    def can_allocate(self, tokens, share=True):
        raise NotImplementedError(
            "TODO 03.9: step 1: say whether allocate would succeed, see the"
            " lesson's 'Steps' section")

    def allocate(self, sid, tokens, share=True):
        """Give sequence `sid` blocks for the prompt `tokens`. Cached blocks
        of a shared start are reused, not recomputed. Returns the number of
        tokens already in the cache (a multiple of the block size)."""
        raise NotImplementedError(
            "TODO 03.9: step 1: give sequence sid blocks for its prompt, "
            "reusing cached ones, changing nothing on failure, and return "
            "the tokens already cached, see the lesson's 'Steps' section")

    def commit(self, sid, tokens, n_done):
        """Say that the first `n_done` tokens of `sid` are now computed. Every
        block they fill completely becomes a cached prefix others can reuse."""
        raise NotImplementedError(
            "TODO 03.9: step 3: register the hashes of blocks that n_done "
            "tokens fill completely, see the lesson's 'Steps' section")

    def extend(self, sid, n=1):
        """Make room for `n` more tokens. Returns the copies to make, a list
        of (from_block, to_block): a block shared with another sequence is
        copied before this one writes into it (copy-on-write)."""
        raise NotImplementedError(
            "TODO 03.9: step 4: make room for n more tokens, copying a "
            "shared last block first, see the lesson's 'Steps' section")

    def fork(self, sid, child):
        """A second sequence that starts as a copy of `sid` (parallel samples,
        beam search): it shares every block until one of them writes."""
        raise NotImplementedError(
            "TODO 03.9: step 4: share every block of sid with child, see "
            "the lesson's 'Steps' section")

    def free(self, sid):
        """Release a sequence. Blocks go back last-first, so a cached prefix is
        evicted from its end, and blocks no one else holds are reusable."""
        raise NotImplementedError(
            "TODO 03.9: step 5: release the sequence's blocks last-first, "
            "see the lesson's 'Steps' section")

    # -- numbers for reports --
    @property
    def hit_rate(self):
        return self.hit_tokens / self.query_tokens if self.query_tokens else 0.0

    def waste_slots(self):
        """Token slots held by blocks in use but holding nothing yet."""
        raise NotImplementedError(
            "TODO 03.9: step 1: count the slots held by blocks in use that "
            "hold nothing yet, see the lesson's 'Steps' section")
