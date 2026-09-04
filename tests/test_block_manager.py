import pytest

from tinyserve.engine.block_manager import BlockManager, compute_hash
from tinyserve.engine.sequence import Sequence


def seq(tokens, id="s"):
    return Sequence(id=id, prompt_tokens=list(tokens))


def test_allocate_and_free_round_trip():
    bm = BlockManager(num_blocks=4, block_size=4)
    s = seq(range(10))  # 3 blocks: 4 + 4 + 2
    assert bm.can_allocate(s)
    bm.allocate(s)
    assert len(s.block_table) == 3
    assert bm.num_free == 1
    bm.free(s)
    assert bm.num_free == 4
    assert s.block_table == []


def test_can_allocate_is_conservative():
    bm = BlockManager(num_blocks=2, block_size=4)
    assert not bm.can_allocate(seq(range(9)))
    assert bm.can_allocate(seq(range(8)))


def test_identical_prefix_shares_full_blocks():
    bm = BlockManager(num_blocks=8, block_size=4)
    a = seq([1, 2, 3, 4, 5, 6, 7, 8, 9], "a")
    b = seq([1, 2, 3, 4, 5, 6, 7, 8, 50], "b")
    bm.allocate(a)
    bm.allocate(b)
    assert a.block_table[:2] == b.block_table[:2]      # shared
    assert a.block_table[2] != b.block_table[2]        # private tails
    assert bm.blocks[a.block_table[0]].ref_count == 2
    assert b.num_cached_tokens == 8
    assert a.num_cached_tokens == 0
    assert bm.num_used == 4


def test_partial_block_is_never_shared():
    bm = BlockManager(num_blocks=8, block_size=4)
    a = seq([1, 2, 3], "a")
    b = seq([1, 2, 3], "b")
    bm.allocate(a)
    bm.allocate(b)
    assert a.block_table != b.block_table
    assert b.num_cached_tokens == 0


def test_fully_cached_prompt_still_prefills_one_token():
    bm = BlockManager(num_blocks=8, block_size=4)
    a = seq([1, 2, 3, 4, 5, 6, 7, 8], "a")
    b = seq([1, 2, 3, 4, 5, 6, 7, 8], "b")
    bm.allocate(a)
    bm.allocate(b)
    assert b.block_table == a.block_table
    assert b.num_cached_tokens == 7


def test_prefix_hash_is_chained():
    assert compute_hash([1, 2], compute_hash([9, 9])) != compute_hash([1, 2], compute_hash([8, 8]))
    bm = BlockManager(num_blocks=8, block_size=2)
    a = seq([9, 9, 1, 2], "a")
    b = seq([8, 8, 1, 2], "b")
    bm.allocate(a)
    bm.allocate(b)
    assert a.block_table[1] != b.block_table[1]


def test_shared_block_survives_until_last_reference_freed():
    bm = BlockManager(num_blocks=4, block_size=2)
    a = seq([1, 2, 3], "a")
    b = seq([1, 2, 4], "b")
    bm.allocate(a)
    bm.allocate(b)
    bm.free(a)
    assert bm.blocks[b.block_table[0]].ref_count == 1
    assert bm.num_free == 2
    bm.free(b)
    assert bm.num_free == 4


def test_freed_block_can_still_be_hit_until_reused():
    bm = BlockManager(num_blocks=3, block_size=2)
    a = seq([1, 2, 3], "a")
    bm.allocate(a)
    bm.free(a)
    b = seq([1, 2, 7], "b")
    bm.allocate(b)
    assert b.num_cached_tokens == 2
    assert bm.cache_hits == 1


def test_may_append_allocates_only_at_block_boundary():
    bm = BlockManager(num_blocks=3, block_size=2)
    s = seq([1, 2, 3])
    bm.allocate(s)                 # blocks: [1,2] [3]
    s.append(4)                    # fills block 1
    assert bm.may_append(s) and len(s.block_table) == 2
    s.append(5)                    # needs block 2
    assert bm.may_append(s) and len(s.block_table) == 3
    s.append(6)
    assert bm.may_append(s) and len(s.block_table) == 3
    s.append(7)
    assert not bm.may_append(s)    # pool exhausted


def test_full_block_is_registered_only_once_its_last_kv_is_written():
    """A token joins the sequence one step before its K/V reaches the
    pool, so a just-filled block must not be shareable until the sequence
    asks for the *next* block. Registering early + a preemption in between
    would hand a later hit a block with an unwritten final slot."""
    bm = BlockManager(num_blocks=4, block_size=2)
    s = seq([1, 2, 3])
    bm.allocate(s)
    s.append(4)                  # block 1 = [3, 4] is full, but 4 is unwritten
    bm.may_append(s)
    assert bm.blocks[s.block_table[1]].hash == -1
    s.append(5)                  # needs block 2 -> block 1 is now written
    bm.may_append(s)
    assert bm.blocks[s.block_table[1]].hash != -1
    t = seq([1, 2, 3, 4, 5], "t")
    bm.allocate(t)
    assert t.num_cached_tokens == 4


def test_generated_block_not_registered_if_sequence_stops_on_boundary():
    bm = BlockManager(num_blocks=4, block_size=2)
    s = seq([1, 2, 3])
    bm.allocate(s)
    s.append(4)
    bm.may_append(s)
    bm.free(s)                   # finished with the last slot never written
    t = seq([1, 2, 3, 4, 5], "t")
    bm.allocate(t)
    assert t.num_cached_tokens == 2   # only the prompt block was reusable


def test_prefix_caching_can_be_disabled():
    bm = BlockManager(num_blocks=8, block_size=2, prefix_caching=False)
    a = seq([1, 2, 3], "a")
    b = seq([1, 2, 3], "b")
    bm.allocate(a)
    bm.allocate(b)
    assert not set(a.block_table) & set(b.block_table)


def test_rejects_zero_blocks():
    with pytest.raises(ValueError):
        BlockManager(0, 16)
