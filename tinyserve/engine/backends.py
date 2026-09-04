"""KV backends: the seam between the scheduler and the cache design.

The scheduler does not know whether KV lives in padded rows or paged
blocks. It asks a backend five things:

    can_admit(seq)        is there room to prefill this sequence?
    admit(seq)            reserve that room
    prefill_caches(seq)   per-layer caches for a B=1 forward over its
                          not-yet-cached tokens
    after_prefill(seq, caches)
    may_append(seq)       make room for the next generated token; False
                          means "preempt someone"
    decode_caches(seqs)   per-layer caches for a batched one-token forward
    free(seq)             give it all back

Both backends get the same byte budget. How much of it turns into served
tokens is the Session 7-8 measurement.
"""

from dataclasses import dataclass

import mlx.core as mx

from tinyserve.engine.block_manager import BlockManager
from tinyserve.engine.kv_cache import KVCache, PaddedBatchKVCache, _round_up
from tinyserve.engine.paged_cache import BatchPlan, PagedKVCache
from tinyserve.engine.sequence import Sequence


@dataclass(frozen=True)
class KVSpec:
    """What one token of KV costs, from the model's config."""

    num_layers: int
    n_kv_heads: int
    head_dim: int
    itemsize: int = 2  # fp16 / bf16 activations

    @property
    def bytes_per_token(self) -> int:
        return self.num_layers * 2 * self.n_kv_heads * self.head_dim * self.itemsize


class PaddedBackend:
    """Sessions 4-6: one left-padded tensor per layer.

    Memory accounting counts what the tensor *reserves* — B rows times the
    longest row, rounded up to the growth step — because that is what the
    machine actually has to hold.
    """

    name = "padded"

    def __init__(self, spec: KVSpec, budget_bytes: int):
        self.spec = spec
        self.budget_tokens = max(1, budget_bytes // spec.bytes_per_token)
        self.caches = [PaddedBatchKVCache() for _ in range(spec.num_layers)]
        self.rows: list[Sequence] = []
        self.copies = 0  # full-tensor copies (admit + evict), for the write-up

    @property
    def capacity_tokens(self) -> int:
        return self.budget_tokens

    def _fits(self, rows: int, length: int) -> bool:
        return rows * length <= self.budget_tokens

    def can_admit(self, seq: Sequence) -> bool:
        c = self.caches[0]
        length = _round_up(max(c._idx, seq.num_tokens) + 1, PaddedBatchKVCache.step)
        return self._fits(len(self.rows) + 1, length)

    def admit(self, seq: Sequence) -> None:
        pass  # nothing to reserve until the prefill produces a row

    def prefill_caches(self, seq: Sequence) -> list:
        seq.num_cached_tokens = 0
        return [KVCache() for _ in range(self.spec.num_layers)]

    def after_prefill(self, seq: Sequence, caches: list) -> None:
        for batch_cache, seq_cache in zip(self.caches, caches):
            batch_cache.extend(seq_cache)
        self.rows.append(seq)
        self.copies += 1

    def may_append(self, seq: Sequence) -> bool:
        c = self.caches[0]
        if c._idx + 1 <= c.capacity:
            return True
        return self._fits(len(self.rows), c.capacity + PaddedBatchKVCache.step)

    def decode_caches(self, seqs: list[Sequence]) -> list:
        if [s.id for s in seqs] != [s.id for s in self.rows]:
            raise RuntimeError("decode batch order does not match cache rows")
        return self.caches

    def free(self, seq: Sequence) -> None:
        i = next(k for k, s in enumerate(self.rows) if s is seq)
        self.rows.pop(i)
        keep = [k for k in range(len(self.rows) + 1) if k != i]
        for c in self.caches:
            c.filter(keep)
        self.copies += 1
        seq.reset_kv_state()

    def stats(self) -> dict:
        c = self.caches[0]
        return {
            "backend": self.name,
            "rows": len(self.rows),
            "reserved_tokens": len(self.rows) * c.capacity,
            "used_tokens": sum(c._idx - p for p in c.left_pad),
            "budget_tokens": self.budget_tokens,
            "kv_bytes": sum(x.nbytes for x in self.caches),
            "copies": self.copies,
        }


class PagedBackend:
    """Sessions 7-9: block manager + slot pools."""

    name = "paged"

    def __init__(self, spec: KVSpec, budget_bytes: int, block_size: int = 16,
                 prefix_caching: bool = True):
        self.spec = spec
        self.block_size = block_size
        num_blocks = budget_bytes // (spec.bytes_per_token * block_size)
        self.bm = BlockManager(num_blocks, block_size, prefix_caching)
        self.caches = [
            PagedKVCache(num_blocks * block_size, spec.n_kv_heads, spec.head_dim)
            for _ in range(spec.num_layers)
        ]

    @property
    def capacity_tokens(self) -> int:
        return self.bm.num_blocks * self.block_size

    def _slot(self, seq: Sequence, pos: int) -> int:
        bs = self.block_size
        return seq.block_table[pos // bs] * bs + pos % bs

    def _set_plan(self, plan: BatchPlan) -> list:
        for c in self.caches:
            c.plan = plan
        return self.caches

    def can_admit(self, seq: Sequence) -> bool:
        return self.bm.can_allocate(seq)

    def admit(self, seq: Sequence) -> None:
        self.bm.allocate(seq)

    def prefill_caches(self, seq: Sequence) -> list:
        n, cached = seq.num_tokens, seq.num_cached_tokens
        slots = [self._slot(seq, p) for p in range(n)]
        plan = BatchPlan(
            slot_table=mx.array([slots], dtype=mx.int32),
            write_slots=mx.array(slots[cached:], dtype=mx.int32),
            offsets=mx.array([cached], dtype=mx.int32),
        )
        return self._set_plan(plan)

    def after_prefill(self, seq: Sequence, caches: list) -> None:
        pass  # the pool already holds the K/V

    def may_append(self, seq: Sequence) -> bool:
        return self.bm.may_append(seq)

    def decode_caches(self, seqs: list[Sequence]) -> list:
        bs = self.block_size
        nb = max(s.num_blocks(bs) for s in seqs)
        tables = [s.block_table + [0] * (nb - len(s.block_table)) for s in seqs]
        table = mx.array(tables, dtype=mx.int32)                        # (B, nb)
        slot_table = (table[:, :, None] * bs + mx.arange(bs, dtype=mx.int32)).reshape(len(seqs), nb * bs)
        plan = BatchPlan(
            slot_table=slot_table,
            write_slots=mx.array([self._slot(s, s.num_tokens - 1) for s in seqs], dtype=mx.int32),
            offsets=mx.array([s.num_tokens - 1 for s in seqs], dtype=mx.int32),
        )
        return self._set_plan(plan)

    def free(self, seq: Sequence) -> None:
        self.bm.free(seq)

    def stats(self) -> dict:
        return {
            "backend": self.name,
            "kv_bytes": sum(c.nbytes for c in self.caches),
            "budget_tokens": self.capacity_tokens,
            **self.bm.stats(),
        }


def make_backend(name: str, spec: KVSpec, budget_bytes: int, block_size: int = 16,
                 prefix_caching: bool = True):
    if name == "paged":
        return PagedBackend(spec, budget_bytes, block_size, prefix_caching)
    if name == "padded":
        return PaddedBackend(spec, budget_bytes)
    raise ValueError(f"unknown kv backend {name!r}")
