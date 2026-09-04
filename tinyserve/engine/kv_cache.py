"""Contiguous KV caches — the design the paged allocator replaces.

* ``KVCache``: one sequence, one growing (1, H, L, D) tensor per layer.
  Sessions 1-3. Simple, and the reference every batched path is diffed
  against token-for-token.

* ``PaddedBatchKVCache``: a batch of sequences in one (B, H, L, D) tensor
  per layer, left-padded so every row ends at the same column. Sessions
  4-6. Admitting a sequence re-pads and copies the whole tensor; evicting
  one copies it again; memory is B x longest-row whether or not the other
  rows are that long. Those three costs are what Session 7 measures away.

Both classes speak the protocol mlx-lm's models expect from a cache:
``offset`` (RoPE position of the next token), ``update_and_fetch(k, v)``
(append, return everything so far) and, when the cache needs to shape
attention itself, ``make_mask(N)``.
"""

import mlx.core as mx


def _round_up(n: int, step: int) -> int:
    return ((n + step - 1) // step) * step


class KVCache:
    """One sequence's K and V for one layer, grown in steps of 256."""

    step = 256

    def __init__(self):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0  # tokens stored so far == RoPE position of the next one

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        prev = self.offset
        n = keys.shape[2]
        if self.keys is None or prev + n > self.keys.shape[2]:
            B, H, _, Dk = keys.shape
            grow = _round_up(n, self.step)
            new_k = mx.zeros((B, H, grow, Dk), keys.dtype)
            new_v = mx.zeros((B, H, grow, values.shape[3]), values.dtype)
            if self.keys is None:
                self.keys, self.values = new_k, new_v
            else:
                self.keys = mx.concatenate([self.keys[..., :prev, :], new_k], axis=2)
                self.values = mx.concatenate([self.values[..., :prev, :], new_v], axis=2)
        self.offset = prev + n
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    # No make_mask: mlx-lm builds the right causal mask itself for a single
    # sequence ("causal" for prefill, None for one-token decode).

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class PaddedBatchKVCache:
    """B sequences in one tensor per layer, left-padded to a common end.

        row 0:  [_ _ t t t t]      left_pad = 2
        row 1:  [t t t t t t]      left_pad = 0
        row 2:  [_ _ _ _ t t]      left_pad = 4
                           ^ _idx: every row's next token lands here

    Left padding (rather than right) is what lets a single decode step
    write column ``_idx`` for every row at once.
    """

    step = 64

    def __init__(self):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self._idx = 0  # columns filled
        self.left_pad: list[int] = []

    @property
    def batch_size(self) -> int:
        return len(self.left_pad)

    @property
    def offset(self) -> mx.array:
        """Per-row RoPE position of the next token == the row's real length."""
        return mx.array([self._idx - p for p in self.left_pad])

    @property
    def capacity(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, n, Dk = keys.shape
        if B != self.batch_size:
            raise ValueError(f"batch has {B} rows, cache has {self.batch_size}")
        prev = self._idx
        if self.keys is None or prev + n > self.keys.shape[2]:
            grow = _round_up(prev + n, self.step)
            new_k = mx.zeros((B, H, grow, Dk), keys.dtype)
            new_v = mx.zeros((B, H, grow, values.shape[3]), values.dtype)
            if self.keys is not None:
                new_k[..., :prev, :] = self.keys[..., :prev, :]
                new_v[..., :prev, :] = self.values[..., :prev, :]
            self.keys, self.values = new_k, new_v
        self._idx = prev + n
        self.keys[..., prev : self._idx, :] = keys
        self.values[..., prev : self._idx, :] = values
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    def make_mask(self, N: int, **_) -> mx.array:
        """Boolean (B, 1, N, L) mask. Called by the model *before* this
        step's update_and_fetch, so the key axis is the current fill plus
        the N tokens about to be appended.

        Query i of row b sits at column _idx + i and may see column j iff
        j is not padding for that row and j <= _idx + i (causal).
        """
        L = self._idx + N
        cols = mx.arange(L)
        qcols = self._idx + mx.arange(N)
        causal = cols[None, :] <= qcols[:, None]                      # (N, L)
        not_pad = cols[None, :] >= mx.array(self.left_pad)[:, None]  # (B, L)
        return (causal[None, :, :] & not_pad[:, None, :])[:, None]   # (B, 1, N, L)

    def extend(self, cache: KVCache) -> None:
        """Admit one sequence: append its cache as a new row.

        Whichever side is shorter gets padded on the left, and when the
        newcomer is the longer one every existing row is re-padded — a copy
        of the entire batch cache per admission.
        """
        P = cache.offset
        k = cache.keys[..., :P, :]
        v = cache.values[..., :P, :]
        if self.keys is None:
            self.keys, self.values = k, v
            self._idx = P
            self.left_pad = [0]
            return

        if P > self._idx:
            delta = P - self._idx
            pad = [(0, 0), (0, 0), (delta, 0), (0, 0)]
            self.keys = mx.pad(self.keys[..., : self._idx, :], pad)
            self.values = mx.pad(self.values[..., : self._idx, :], pad)
            self.left_pad = [p + delta for p in self.left_pad]
            self._idx = P
            left = 0
        else:
            left = self._idx - P
            if left:
                pad = [(0, 0), (0, 0), (left, 0), (0, 0)]
                k, v = mx.pad(k, pad), mx.pad(v, pad)
            self.keys = self.keys[..., : self._idx, :]
            self.values = self.values[..., : self._idx, :]

        self.keys = mx.concatenate([self.keys, k], axis=0)
        self.values = mx.concatenate([self.values, v], axis=0)
        self.left_pad.append(left)

    def filter(self, keep: list[int]) -> None:
        """Evict: keep only the given rows (another full copy), then drop
        any padding columns that every survivor shares."""
        if not keep:
            self.__init__()
            return
        idx = mx.array(keep)
        self.keys = self.keys[idx]
        self.values = self.values[idx]
        self.left_pad = [self.left_pad[i] for i in keep]
        common = min(self.left_pad)
        if common:
            self.keys = self.keys[..., common:, :]
            self.values = self.values[..., common:, :]
            self._idx -= common
            self.left_pad = [p - common for p in self.left_pad]

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes
