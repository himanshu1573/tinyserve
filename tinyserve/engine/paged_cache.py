"""The paged KV cache as MLX sees it.

The block manager hands out block ids; this module turns them into reads
and writes on one flat pool per layer:

    pool: (num_slots, n_kv_heads, head_dim),  slot = block_id * block_size + i

There is no custom attention kernel on this machine, so PagedAttention is
done the honest way: before each forward the backend builds a ``BatchPlan``
naming, for every row, the slots to write this step's K/V into and the
slots to gather the whole context from. The gather is a copy of the
context per layer per step — the price of paging without a kernel, and
one of the numbers the write-up reports.

A ``PagedKVCache`` lives per layer for the life of the engine and speaks
the same protocol as ``KVCache``, so the unmodified mlx-lm model runs on
top of it. ``offset`` is a per-row array — mx.fast.rope accepts one — and
``make_mask`` is what lets rows of different lengths share one forward.
"""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class BatchPlan:
    """Everything a forward needs to know about where its KV lives.

    slot_table:  (B, L) int32 — physical slot of key position j for row b.
                 Padded past a row's length with any valid slot; masked out.
    write_slots: (B*N,) int32 — where this step's N new tokens per row go,
                 flattened row-major to match keys.transpose(0,2,1,3).
    offsets:     (B,) int32 — RoPE position of each row's first new token.
    """

    slot_table: mx.array
    write_slots: mx.array
    offsets: mx.array

    def mask(self, N: int) -> mx.array:
        """Boolean (B, 1, N, L): query i of row b is at position
        offsets[b] + i and may see key position j iff j <= that."""
        L = self.slot_table.shape[1]
        j = mx.arange(L)[None, None, None, :]
        q = (self.offsets[:, None] + mx.arange(N)[None, :])[:, None, :, None]
        return j <= q


class PagedKVCache:
    """One layer's slot pool. Allocated lazily on the first write so the
    dtype follows the model's activations."""

    def __init__(self, num_slots: int, n_kv_heads: int, head_dim: int):
        self.num_slots = num_slots
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.plan: BatchPlan | None = None

    @property
    def offset(self) -> mx.array:
        return self.plan.offsets

    def make_mask(self, N: int, **_) -> mx.array:
        return self.plan.mask(N)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if self.plan is None:
            raise RuntimeError("PagedKVCache used without a BatchPlan")
        B, H, N, Dk = keys.shape
        Dv = values.shape[3]
        if self.keys is None:
            self.keys = mx.zeros((self.num_slots, H, Dk), keys.dtype)
            self.values = mx.zeros((self.num_slots, H, Dv), values.dtype)

        # Write: (B, H, N, D) -> (B*N, H, D), one row per new token.
        self.keys[self.plan.write_slots] = keys.transpose(0, 2, 1, 3).reshape(B * N, H, Dk)
        self.values[self.plan.write_slots] = values.transpose(0, 2, 1, 3).reshape(B * N, H, Dv)

        # Read: gather every row's context, (B, L, H, D) -> (B, H, L, D).
        k = self.keys[self.plan.slot_table].transpose(0, 2, 1, 3)
        v = self.values[self.plan.slot_table].transpose(0, 2, 1, 3)
        return k, v

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes
