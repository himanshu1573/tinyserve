"""Paged KV bookkeeping: a free list, per-sequence block tables, refcounts.

Pure Python. No MLX here — this module decides *where* each token's K/V
lives; ``paged_cache.py`` does the reading and writing.

The idea from the vLLM paper (§4): the KV cache is carved into fixed
blocks of ``block_size`` tokens. A sequence owns a *block table* mapping
its logical blocks (0, 1, 2, ...) to physical ones (any order, anywhere in
the pool). Growth allocates one block at a time, so the only waste is the
tail of the last block — never a padded row, never a reserved span.

Prefix sharing (Session 9) falls out of refcounts: a full block's content
is identified by a chained hash of every token up to and including it.
Two sequences with the same first 32 tokens map their first two logical
blocks to the same two physical blocks and each holds a reference. Full
blocks are immutable, so no copy-on-write is ever needed; the partial
last block is always private.
"""

from collections import deque
from dataclasses import dataclass, field

from tinyserve.engine.sequence import Sequence


@dataclass
class Block:
    id: int
    ref_count: int = 0
    hash: int = -1  # -1: not a full, shareable block
    token_ids: list[int] = field(default_factory=list)

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


def compute_hash(token_ids: list[int], prefix_hash: int = -1) -> int:
    """Chained hash: a block's identity includes everything before it, so
    identical tokens after different prefixes never collide."""
    return hash((prefix_hash, tuple(token_ids)))


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, prefix_caching: bool = True):
        if num_blocks < 1:
            raise ValueError("KV budget too small for a single block")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.prefix_caching = prefix_caching
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.hash_to_block: dict[int, int] = {}
        # Counters for the write-up.
        self.cache_hits = 0
        self.cache_misses = 0

    # --- capacity ------------------------------------------------------------

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    def can_allocate(self, seq: Sequence) -> bool:
        """Conservative: assumes no prefix hits. A sequence that fits this
        way always fits; one that doesn't might still fit thanks to sharing,
        and we prefer to wait a step over promising memory we may not have."""
        return seq.num_blocks(self.block_size) <= self.num_free

    # --- allocate / grow / free ---------------------------------------------

    def _take_free_block(self) -> Block:
        block = self.blocks[self.free_blocks.popleft()]
        if block.hash != -1:
            # A freed block keeps its identity while it sits in the free list
            # so a later request can still hit it. Reusing it for new
            # content ends that.
            self.hash_to_block.pop(block.hash, None)
        block.reset()
        return block

    def allocate(self, seq: Sequence) -> None:
        """Give a WAITING sequence blocks for all of its current tokens,
        reusing shared prefix blocks where the hashes match."""
        if seq.block_table:
            raise RuntimeError(f"{seq.id} already has a block table")
        bs = self.block_size
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks(bs)):
            toks = seq.block_tokens(i, bs)
            full = len(toks) == bs
            h = compute_hash(toks, h) if (full and self.prefix_caching) else -1

            block_id = self.hash_to_block.get(h, -1) if h != -1 else -1
            hit = (
                block_id != -1
                and self.blocks[block_id].token_ids == toks
            )
            if not hit:
                cache_miss = True
                block = self._take_free_block()
                block_id = block.id
                if h != -1:
                    block.hash = h
                    block.token_ids = toks
                    self.hash_to_block[h] = block_id
                if full:
                    self.cache_misses += 1
            else:
                block = self.blocks[block_id]
                if block.ref_count == 0:
                    self.free_blocks.remove(block_id)
                block.ref_count += 1
                self.cache_hits += 1
                if not cache_miss:
                    seq.num_cached_tokens += bs
            seq.block_table.append(block_id)

        # Prefill needs at least one token to produce logits from. If the
        # whole prompt was cached, recompute the last token: its K/V lands
        # in the slot that already holds identical values.
        seq.num_cached_tokens = min(seq.num_cached_tokens, seq.num_tokens - 1)

    def may_append(self, seq: Sequence) -> bool:
        """Make sure the sequence's newest token has a slot. Allocates a
        block when the previous one just filled; returns False if the pool
        is empty, which is the scheduler's cue to preempt someone."""
        if seq.num_blocks(self.block_size) <= len(seq.block_table):
            return True
        if not self.free_blocks:
            return False
        # Only now is the previous block complete *in the pool*. A token is
        # appended to the sequence one step before its K/V is written (the
        # write happens when it is fed to the next forward), so a block
        # whose last token was just appended is not yet shareable. If we
        # registered it here and the sequence got preempted before that
        # write, a later hit would read zeros for its final position.
        self._register_full_block(seq, len(seq.block_table) - 1)
        seq.block_table.append(self._take_free_block().id)
        return True

    def _register_full_block(self, seq: Sequence, i: int) -> None:
        """Offer logical block i — full and fully written — for sharing."""
        if not self.prefix_caching:
            return
        block = self.blocks[seq.block_table[i]]
        if block.hash != -1 or block.ref_count != 1:
            return  # already shareable (hits always carry a hash), or shared
        prev_hash = self.blocks[seq.block_table[i - 1]].hash if i > 0 else -1
        toks = seq.block_tokens(i, self.block_size)
        if len(toks) != self.block_size or (i > 0 and prev_hash == -1):
            return
        block.hash = compute_hash(toks, prev_hash)
        block.token_ids = toks
        self.hash_to_block[block.hash] = block.id

    def free(self, seq: Sequence) -> None:
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free_blocks.append(block_id)
        seq.reset_kv_state()

    def stats(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "used_blocks": self.num_used,
            "free_blocks": self.num_free,
            "prefix_hits": self.cache_hits,
            "prefix_misses": self.cache_misses,
        }
