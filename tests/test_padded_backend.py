from tinyserve.engine.backends import KVSpec, PaddedBackend
from tinyserve.engine.kv_cache import PaddedBatchKVCache
from tinyserve.engine.sequence import Sequence
from tests.fakes import FakeRunner

SPEC = KVSpec(num_layers=2, n_kv_heads=1, head_dim=4)


def prefilled(backend, runner, seq):
    backend.admit(seq)
    caches = backend.prefill_caches(seq)
    runner.prefill(seq.all_tokens(), caches)
    backend.after_prefill(seq, caches)


def test_admit_budget_counts_reserved_not_used():
    step = PaddedBatchKVCache.step
    backend = PaddedBackend(SPEC, budget_bytes=SPEC.bytes_per_token * step * 2)
    runner = FakeRunner()
    a = Sequence(id="a", prompt_tokens=[1, 2, 3])
    assert backend.can_admit(a)
    prefilled(backend, runner, a)
    b = Sequence(id="b", prompt_tokens=[1, 2, 3])
    assert backend.can_admit(b)
    prefilled(backend, runner, b)
    c = Sequence(id="c", prompt_tokens=[1])
    assert not backend.can_admit(c)   # 3 rows x 64 reserved > 128 budget
    backend.free(a)
    assert backend.can_admit(c)


def test_decode_rows_track_admissions_and_evictions():
    backend = PaddedBackend(SPEC, budget_bytes=10**9)
    runner = FakeRunner()
    seqs = [Sequence(id=f"s{i}", prompt_tokens=[1] * (i + 1)) for i in range(3)]
    for s in seqs:
        prefilled(backend, runner, s)
    assert backend.caches[0].left_pad == [2, 1, 0]
    backend.free(seqs[1])
    assert [s.id for s in backend.rows] == ["s0", "s2"]
    assert backend.caches[0].left_pad == [2, 0]
    for s in (seqs[0], seqs[2]):
        s.append(5)
        assert backend.may_append(s)
    runner.decode_batch([5, 5], backend.decode_caches([seqs[0], seqs[2]]))
    assert backend.caches[0].offset.tolist() == [2, 4]
    assert backend.stats()["copies"] == 4
