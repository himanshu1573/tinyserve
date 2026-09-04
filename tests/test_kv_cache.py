import mlx.core as mx

from tinyserve.engine.kv_cache import KVCache, PaddedBatchKVCache

H, D = 2, 4


def filled(n, value):
    """A KVCache holding n positions whose entries equal value + position."""
    c = KVCache()
    ks = mx.arange(n, dtype=mx.float32)[None, None, :, None] + value
    ks = mx.broadcast_to(ks, (1, H, n, D))
    c.update_and_fetch(ks, ks)
    return c


def test_kvcache_grows_and_returns_only_filled_positions():
    c = KVCache()
    k, v = c.update_and_fetch(mx.ones((1, H, 3, D)), mx.ones((1, H, 3, D)))
    assert k.shape == (1, H, 3, D) and c.offset == 3
    c.step = 4
    k, _ = c.update_and_fetch(mx.ones((1, H, 2, D)) * 2, mx.ones((1, H, 2, D)))
    assert k.shape == (1, H, 5, D)
    assert k[0, 0, 4, 0].item() == 2 and k[0, 0, 2, 0].item() == 1
    assert c.nbytes > 0


def test_padded_extend_aligns_rows_on_the_right():
    b = PaddedBatchKVCache()
    b.extend(filled(3, 100))
    b.extend(filled(5, 200))
    b.extend(filled(2, 300))
    assert b.left_pad == [2, 0, 3] and b._idx == 5
    assert b.offset.tolist() == [3, 5, 2]
    # Row 0's real data ends at column 4 with value 100 + 2.
    assert b.keys[0, 0, 4, 0].item() == 102
    assert b.keys[2, 0, 3, 0].item() == 300
    assert b.keys[2, 0, 2, 0].item() == 0  # padding


def test_padded_update_writes_the_same_column_for_every_row():
    b = PaddedBatchKVCache()
    b.extend(filled(3, 100))
    b.extend(filled(5, 200))
    new = mx.ones((2, H, 1, D)) * mx.array([7.0, 8.0])[:, None, None, None]
    k, _ = b.update_and_fetch(new, new)
    assert k.shape == (2, H, 6, D)
    assert k[0, 0, 5, 0].item() == 7 and k[1, 0, 5, 0].item() == 8
    assert b.offset.tolist() == [4, 6]


def test_padded_mask_hides_padding_and_future():
    b = PaddedBatchKVCache()
    b.extend(filled(1, 0))
    b.extend(filled(3, 0))
    m = b.make_mask(1)             # one new token per row, key axis = 4
    assert m.shape == (2, 1, 1, 4)
    assert m[0, 0, 0].tolist() == [False, False, True, True]
    assert m[1, 0, 0].tolist() == [True, True, True, True]


def test_padded_filter_drops_rows_and_shared_padding():
    b = PaddedBatchKVCache()
    b.extend(filled(2, 100))
    b.extend(filled(6, 200))
    b.extend(filled(4, 300))
    b.filter([0, 2])
    assert b.left_pad == [2, 0] and b._idx == 4
    assert b.keys.shape[0] == 2
    assert b.keys[0, 0, 3, 0].item() == 101
    assert b.keys[1, 0, 3, 0].item() == 303
    b.filter([])
    assert b.keys is None and b.batch_size == 0
