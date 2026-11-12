"""Module 03: the KV cache, then the block allocator with prefix sharing.

    from servinglab.kv import KVCache, generate_cached               # lesson 03.2
    from servinglab.kv import BlockAllocator, PagedCache             # 03.6, 03.7
    from servinglab.kv import trace_tokens, replay                   # 03.9

The cache wraps the engine's own steps (`rmsnorm`, `rope`, `attention`):
a call computes only the NEW tokens and reads the old keys and values from
memory. Everything here is plain NumPy and keeps the engine's shapes:

    k, v of one layer   (positions, kv_heads, head_dim)

Two homes for those arrays share one small interface (`append`, `advance`,
`length`), so one forward pass serves both:

    KVCache       one contiguous array per sequence (03.2): simple, wasteful.
    PagedCache    a pool of fixed-size blocks and a block table per sequence
                  (03.6): what vLLM calls PagedAttention, in NumPy.

`BlockAllocator` is the bookkeeping alone (no arrays), so the scheduler in
Module 04 can use it in a simulator that holds no tensors.

Shortcuts, each with its limit: block hashes use Python's tuple hash (a
collision would serve wrong keys; vLLM can use SHA-256 for this), and
`PagedCache` gathers a sequence's blocks into one array on every read (a real
kernel reads the blocks in place).
"""

from array import array
from collections import OrderedDict
from dataclasses import dataclass, field
import heapq
import math

import numpy as np

from .engine.model import attention, rmsnorm, rope, silu
from .engine.sampling import sample


# ---- 03.2: the cache and the cached forward pass ----------------------------

class KVCache:
    """Keys and values of every layer for ONE sequence, in one array each.

    Room for `max_len` positions is reserved up front (the waste lesson 03.6
    removes). `append(layer, k, v)` writes the new positions of one layer and
    returns all keys and values so far; `advance(n)` moves the length on once
    every layer has been written.
    """

    def __init__(self, config, max_len):
        shape = (config.layers, max_len, config.kv_heads, config.head_dim)
        self.k = np.zeros(shape, np.float32)
        self.v = np.zeros(shape, np.float32)
        self.length = 0

    def append(self, layer, k, v):
        a, b = self.length, self.length + len(k)
        if b > self.k.shape[1]:
            raise ValueError(f"cache full: {b} positions, room for {self.k.shape[1]}")
        self.k[layer, a:b], self.v[layer, a:b] = k, v
        return self.k[layer, :b], self.v[layer, :b]

    def advance(self, n):
        self.length += n

    @property
    def nbytes(self):
        return self.k.nbytes + self.v.nbytes


def step_forward(model, items, last_only=True):
    """One pass over several sequences' new tokens at once.

    `items` is a list of (cache, new_ids): a sequence's cache (anything with
    `length`, `append` and `advance`) and the token ids it has not run yet. A
    prompt, a chunk of one or a single decode token all look the same. All the
    new tokens go through each weight matrix together (the point of batching),
    while attention reads each sequence's own cache.

    Returns one row of logits per item (the next token after its last new
    token), or with last_only=False a list of (len(new_ids), vocab) arrays.
    """
    c, w = model.config, model.weights
    ids = np.concatenate([np.asarray(i, dtype=np.int64) for _, i in items])
    sizes = [len(i) for _, i in items]
    positions = np.concatenate([cache.length + np.arange(n)
                                for (cache, _), n in zip(items, sizes)])
    cuts = np.cumsum(sizes)[:-1]
    x = w["model.embed_tokens.weight"][ids]
    for i in range(c.layers):
        p = f"model.layers.{i}."
        h = rmsnorm(x, w[p + "input_layernorm.weight"], c.eps)
        T = len(h)
        q = (h @ w[p + "self_attn.q_proj.weight"].T).reshape(T, c.heads, c.head_dim)
        k = (h @ w[p + "self_attn.k_proj.weight"].T).reshape(T, c.kv_heads, c.head_dim)
        v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(T, c.kv_heads, c.head_dim)
        q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
        outs = []
        for (cache, _), qs, ks, vs in zip(items, np.split(q, cuts),
                                          np.split(k, cuts), np.split(v, cuts)):
            keys, values = cache.append(i, ks, vs)       # the cache step
            outs.append(attention(qs, keys, values))
        x = x + np.concatenate(outs) @ w[p + "self_attn.o_proj.weight"].T
        h = rmsnorm(x, w[p + "post_attention_layernorm.weight"], c.eps)
        gate = silu(h @ w[p + "mlp.gate_proj.weight"].T)
        x = x + (gate * (h @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T
    for (cache, _), n in zip(items, sizes):
        cache.advance(n)
    head = w["model.embed_tokens.weight" if c.tie_embeddings else "lm_head.weight"]
    if last_only:
        x = x[np.cumsum(sizes) - 1]
        return rmsnorm(x, w["model.norm.weight"], c.eps) @ head.T
    x = rmsnorm(x, w["model.norm.weight"], c.eps) @ head.T
    return np.split(x, cuts)


def forward_cached(model, ids, cache):
    """Logits for the next token, running only `ids` (the tokens the cache
    has not seen). One sequence; `step_forward` does the work."""
    return step_forward(model, [(cache, ids)])[0]


def generate_cached(model, prompt_ids, max_new_tokens=64, seed=None, **sampling):
    """`engine.generate` with a cache: same tokens, far less work.

    The prompt goes through once (prefill); then each step feeds only the
    newest token (decode). `sampling` is passed to `sample`.
    """
    rng = np.random.default_rng(seed)
    cache = KVCache(model.config, len(prompt_ids) + max_new_tokens)
    ids, new = list(prompt_ids), list(prompt_ids)
    for _ in range(max_new_tokens):
        logits = forward_cached(model, new, cache)
        token = sample(logits, history=ids[len(prompt_ids):], rng=rng, **sampling)
        if token in model.config.eos:
            return
        ids.append(token)
        new = [token]
        yield token


def kv_bytes_per_token(config, bytes_per_value=2):
    """Bytes of keys and values one token adds, over all layers (lesson 03.3):
    2 (K and V) x layers x kv_heads x head_dim x bytes per number."""
    return 2 * config.layers * config.kv_heads * config.head_dim * bytes_per_value


def kv_bytes_per_token_from_json(cfg, bytes_per_value=2):
    """Same count from a raw config.json dict (lesson 03.3). Plain
    multi-head or grouped-query attention only: DeepSeek's latent attention
    caches something else, which lesson 03.4 counts."""
    heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", heads)
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
    return 2 * cfg["num_hidden_layers"] * kv_heads * head_dim * bytes_per_value


def memory_budget(model, hardware, utilization=0.9, bytes_per_param=2,
                  kv_bytes_per_value=2, max_batched_tokens=8192,
                  overhead_gb=1.0, block_size=16):
    """Where a GPU's memory goes, and how many tokens of cache are left (03.5).

    weights      every parameter x bytes per parameter
    activations  the biggest tensors alive in one layer for a step of
                 `max_batched_tokens`: about 4 x hidden + 3 x intermediate
                 numbers of 2 bytes per token (an estimate: an engine
                 measures it by running a dummy step at startup)
    overhead     the CUDA context and libraries (`overhead_gb`, an assumption)
    kv           what is left of `utilization` x the GPU's memory

    Returns bytes for each, plus `kv_tokens` and `blocks`.
    """
    from .data import hardware as _hw, model_config as _cfg
    from .roofline import shape
    cfg = _cfg(model) if isinstance(model, str) else model
    chip = _hw(hardware) if isinstance(hardware, str) else hardware
    s = shape(cfg)
    weights = s["params"] * bytes_per_param
    activations = max_batched_tokens * (4 * cfg["hidden_size"] + 3 * cfg["intermediate_size"]) * 2
    usable = chip["memory_gb"] * 1e9 * utilization
    kv = usable - weights - activations - overhead_gb * 1e9
    per_token = kv_bytes_per_token_from_json(cfg, kv_bytes_per_value)
    tokens = max(0, int(kv // per_token))
    return {"total": chip["memory_gb"] * 1e9, "usable": usable, "weights": weights,
            "activations": activations, "overhead": overhead_gb * 1e9, "kv": kv,
            "kv_bytes_per_token": per_token, "kv_tokens": tokens,
            "blocks": tokens // block_size}


# ---- 03.6, 03.7, 03.9: the block allocator -----------------------------------

class OutOfBlocks(Exception):
    """No free block, and none that may be evicted."""


@dataclass
class Seq:
    table: list = field(default_factory=list)   # block ids, in order
    n_tokens: int = 0                           # tokens stored (or reserved)
    n_committed: int = 0                        # tokens whose blocks hold real data
    chain: list = field(default_factory=list)   # hashes of the committed full blocks


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


class PagedView:
    """One sequence's window onto a `PagedCache`, with the cache interface."""

    def __init__(self, cache, sid):
        self.cache, self.sid = cache, sid

    @property
    def length(self):
        return self.cache.computed[self.sid]

    def append(self, layer, k, v):
        return self.cache.write(self.sid, layer, self.length, k, v)

    def advance(self, n):
        self.cache.computed[self.sid] += n


class PagedCache:
    """Real keys and values in a pool of blocks, owned through a `BlockAllocator`.

        cache = PagedCache(model.config, num_blocks=64, block_size=16)
        view, skipped = cache.start("a", prompt_ids)   # skipped: tokens found cached
        logits = step_forward(model, [(view, prompt_ids[skipped:])])[0]
        cache.commit("a", prompt_ids)                  # now others can reuse it
        cache.grow("a", 1)                             # room for one decode token
        cache.finish("a")
    """

    def __init__(self, config, num_blocks, block_size=16):
        self.alloc = BlockAllocator(num_blocks, block_size)
        shape = (config.layers, num_blocks, block_size, config.kv_heads, config.head_dim)
        self.k, self.v = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        self.computed = {}

    def start(self, sid, prompt_ids, share=True):
        skipped = self.alloc.allocate(sid, list(prompt_ids), share)
        self.computed[sid] = skipped
        return PagedView(self, sid), skipped

    def commit(self, sid, ids):
        self.alloc.commit(sid, list(ids), self.computed[sid])

    def grow(self, sid, n=1):
        for old, new in self.alloc.extend(sid, n):
            self.k[:, new], self.v[:, new] = self.k[:, old], self.v[:, old]

    def finish(self, sid):
        self.alloc.free(sid)
        del self.computed[sid]

    def write(self, sid, layer, start, k, v):
        bs, table = self.alloc.block_size, self.alloc.seqs[sid].table
        for i in range(len(k)):
            p = start + i
            b, o = table[p // bs], p % bs
            self.k[layer, b, o], self.v[layer, b, o] = k[i], v[i]
        n = start + len(k)
        blocks = table[:math.ceil(n / bs)]
        shape = (-1,) + self.k.shape[3:]
        return (self.k[layer][blocks].reshape(shape)[:n],
                self.v[layer][blocks].reshape(shape)[:n])


# ---- 03.9: replay a trace through three allocation policies ------------------

def trace_tokens(row, _streams={}):
    """Invented token ids for a trace row (an `array` of ints), so equal prefixes really are equal.

    Traces hold lengths, not text. A request's tokens are the stream named by
    its `prefix_id` for its first `prefix_tokens`, then its own stream. A
    Kodekart task (`kk-task-NN`) is one growing stream whose first 3,000
    tokens are the shared system prompt, so each step extends the last
    (the previous step's reply is not modelled: the trace's own prefix_tokens
    count is what each step may reuse).
    """
    def stream(name, a, b):
        sid = _streams.setdefault(name, len(_streams) + 1)
        return range(sid * 1_000_000 + a, sid * 1_000_000 + b)

    n, p, pid = row["input_tokens"], row["prefix_tokens"], row["prefix_id"]
    if pid.startswith("kk-task"):
        parts = (stream("kk-system", 0, 3000), stream(pid, 3000, n))
    else:
        parts = (stream(pid, 0, p), stream(row["id"], p, n))
    return array("q", (t for part in parts for t in part))


def replay(rows, num_blocks, mode="shared", block_size=16, max_len=None,
           seconds_per_token=0.03):
    """Run requests through an allocation policy and report waste and hits.

    Each request arrives at `arrival_s`, takes memory for its prompt, grows by
    one token every `seconds_per_token` for `output_tokens` steps, then frees
    it. (A fixed pace, not a scheduler: Module 04 builds the real one.) A
    request that finds no memory is turned away and counted.

    mode  "reserve"  one contiguous slab of `max_len` slots per request;
          "paged"    blocks on demand, no sharing;
          "shared"   blocks on demand, with prefix sharing and eviction.

    Returns a dict: admitted, rejected, hit_rate (prompt tokens found cached),
    mean_waste (idle slots / slots held, sampled at each arrival), peak_slots.
    """
    if max_len is None:
        max_len = 1 << math.ceil(math.log2(max(r["input_tokens"] + r["output_tokens"]
                                               for r in rows)))
    alloc = BlockAllocator(num_blocks, block_size)
    slots_total = num_blocks * block_size
    events, admitted, rejected, wastes, peak, held, reserved = [], 0, 0, [], 0, {}, 0

    def run_events(until):
        nonlocal reserved
        while events and events[0][0] <= until:
            t, _, kind, sid = heapq.heappop(events)
            if kind == "grow":
                s = alloc.seqs.get(sid)
                if s and sid in held:
                    try:
                        alloc.extend(sid, min(block_size, held[sid] - s.n_tokens))
                    except OutOfBlocks:
                        pass                          # shortcut: a real engine would preempt
            else:
                if mode == "reserve":
                    reserved -= max_len
                    held.pop(sid)
                else:
                    alloc.free(sid)
                    held.pop(sid, None)

    for i, r in enumerate(rows):
        run_events(r["arrival_s"])
        tokens = trace_tokens(r) if mode != "reserve" else None
        total = r["input_tokens"] + r["output_tokens"]
        if mode == "reserve":
            if reserved + max_len > slots_total:
                rejected += 1
                continue
            reserved += max_len
            alloc.query_tokens += r["input_tokens"]
            held[i] = total
            used = sum(held.values())
            wastes.append(1 - used / reserved)
            peak = max(peak, reserved)
        else:
            try:
                alloc.allocate(i, tokens, share=(mode == "shared"))
            except OutOfBlocks:
                rejected += 1
                continue
            alloc.commit(i, tokens, len(tokens))
            held[i] = total
            slots = alloc.used_blocks * block_size
            wastes.append(alloc.waste_slots() / slots)
            peak = max(peak, slots)
        admitted += 1
        t0 = r["arrival_s"]
        for k in range(1, math.ceil(r["output_tokens"] / block_size) + 1):
            heapq.heappush(events, (t0 + k * block_size * seconds_per_token, i, "grow", i))
        heapq.heappush(events, (t0 + (r["output_tokens"] + 1) * seconds_per_token, i, "done", i))
    return {"mode": mode, "admitted": admitted, "rejected": rejected,
            "hit_rate": alloc.hit_rate, "mean_waste": float(np.mean(wastes)) if wastes else 0.0,
            "peak_slots": peak}
