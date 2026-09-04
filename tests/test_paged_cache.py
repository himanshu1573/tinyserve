import mlx.core as mx

from tinyserve.engine.backends import KVSpec, PagedBackend
from tinyserve.engine.paged_cache import BatchPlan, PagedKVCache
from tinyserve.engine.sequence import Sequence

SPEC = KVSpec(num_layers=1, n_kv_heads=1, head_dim=2)


def test_plan_mask_is_causal_per_row():
    plan = BatchPlan(slot_table=mx.zeros((2, 5), dtype=mx.int32),
                     write_slots=mx.zeros((2,), dtype=mx.int32),
                     offsets=mx.array([1, 3]))
    m = plan.mask(1)
    assert m.shape == (2, 1, 1, 5)
    assert m[0, 0, 0].tolist() == [True, True, False, False, False]
    assert m[1, 0, 0].tolist() == [True, True, True, True, False]
    m = plan.mask(2)  # prefill of 2 tokens at offsets 1 and 3
    assert m[0, 0, 1].tolist() == [True, True, True, False, False]


def test_paged_cache_writes_and_gathers_by_slot():
    cache = PagedKVCache(num_slots=8, n_kv_heads=1, head_dim=2)
    # Two rows: row 0 lives in slots [5, 6], row 1 in slots [2].
    cache.plan = BatchPlan(
        slot_table=mx.array([[5, 6], [2, 0]], dtype=mx.int32),
        write_slots=mx.array([6, 2], dtype=mx.int32),
        offsets=mx.array([1, 0]),
    )
    new = mx.array([[[[6.0, 6.0]]], [[[2.0, 2.0]]]])  # (B=2, H=1, N=1, D=2)
    k, v = cache.update_and_fetch(new, new)
    assert k.shape == (2, 1, 2, 2)
    assert k[0, 0, 1].tolist() == [6.0, 6.0]   # row 0, position 1 -> slot 6
    assert k[1, 0, 0].tolist() == [2.0, 2.0]   # row 1, position 0 -> slot 2
    assert cache.keys[5].tolist() == [[0.0, 0.0]]  # untouched slot


def test_backend_prefill_then_decode_plans():
    backend = PagedBackend(SPEC, budget_bytes=SPEC.bytes_per_token * 4 * 4, block_size=4)
    seq = Sequence(id="s", prompt_tokens=[1, 2, 3, 4, 5])
    assert backend.can_admit(seq)
    backend.admit(seq)
    assert len(seq.block_table) == 2

    caches = backend.prefill_caches(seq)
    plan = caches[0].plan
    b0, b1 = seq.block_table
    assert plan.slot_table.tolist() == [[b0 * 4, b0 * 4 + 1, b0 * 4 + 2, b0 * 4 + 3, b1 * 4]]
    assert plan.write_slots.tolist() == plan.slot_table[0].tolist()
    assert plan.offsets.tolist() == [0]

    seq.append(9)
    assert backend.may_append(seq)
    caches = backend.decode_caches([seq])
    plan = caches[0].plan
    assert plan.write_slots.tolist() == [b1 * 4 + 1]
    assert plan.offsets.tolist() == [5]
    assert plan.slot_table.shape == (1, 8)


def test_backend_uses_cached_prefix_in_prefill_plan():
    backend = PagedBackend(SPEC, budget_bytes=SPEC.bytes_per_token * 4 * 8, block_size=4)
    a = Sequence(id="a", prompt_tokens=[1, 2, 3, 4, 5])
    b = Sequence(id="b", prompt_tokens=[1, 2, 3, 4, 6])
    backend.admit(a)
    backend.admit(b)
    assert b.num_cached_tokens == 4
    plan = backend.prefill_caches(b)[0].plan
    assert plan.write_slots.shape == (1,)          # only the uncached token
    assert plan.offsets.tolist() == [4]
    assert plan.slot_table.shape == (1, 5)


def test_backend_capacity_and_stats():
    backend = PagedBackend(SPEC, budget_bytes=SPEC.bytes_per_token * 4 * 3, block_size=4)
    assert backend.capacity_tokens == 12
    assert backend.stats()["free_blocks"] == 3
