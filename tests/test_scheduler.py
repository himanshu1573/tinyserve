"""Scheduler tests with the fake runner, both KV backends."""

import pytest

from tinyserve.engine.backends import PaddedBackend, PagedBackend
from tinyserve.engine.scheduler import Scheduler
from tinyserve.engine.sequence import Sequence, SeqStatus
from tests.fakes import FakeRunner, expected_continuation


def make_backend(kind, runner, tokens_budget=4096, block_size=4):
    budget = runner.kv_spec.bytes_per_token * tokens_budget
    if kind == "paged":
        return PagedBackend(runner.kv_spec, budget, block_size=block_size)
    return PaddedBackend(runner.kv_spec, budget)


class Recorder:
    def __init__(self):
        self.tokens: dict[str, list[int]] = {}
        self.finished: list[str] = []

    def on_token(self, seq, tok):
        self.tokens.setdefault(seq.id, []).append(tok)

    def on_finish(self, seq):
        self.finished.append(seq.id)


def run_to_completion(sched, limit=500):
    for _ in range(limit):
        if not sched.has_work():
            return
        sched.step()
    raise AssertionError("scheduler did not drain")


def check(seq, rec):
    expected, reason = expected_continuation(seq.prompt_tokens, seq.max_tokens)
    got = [t for t in rec.tokens[seq.id] if t != 999]
    assert got == expected, seq.id
    assert seq.stop_reason == reason
    assert seq.status is SeqStatus.FINISHED


@pytest.mark.parametrize("kind", ["paged", "padded"])
def test_batch_of_different_prompts_generates_each_correctly(kind):
    runner, rec = FakeRunner(), Recorder()
    sched = Scheduler(runner, make_backend(kind, runner), max_batch_size=8,
                      on_token=rec.on_token, on_finish=rec.on_finish)
    seqs = [Sequence(id=f"s{i}", prompt_tokens=list(range(1, 2 + i)), max_tokens=20)
            for i in range(6)]
    for s in seqs:
        sched.add(s)
    run_to_completion(sched)
    for s in seqs:
        check(s, rec)
    assert sorted(rec.finished) == sorted(s.id for s in seqs)
    assert max(runner.batch_sizes) == 6
    assert sched.stats()["running"] == 0


@pytest.mark.parametrize("kind", ["paged", "padded"])
def test_max_batch_size_is_respected(kind):
    runner, rec = FakeRunner(), Recorder()
    sched = Scheduler(runner, make_backend(kind, runner), max_batch_size=2,
                      on_token=rec.on_token, on_finish=rec.on_finish)
    seqs = [Sequence(id=f"s{i}", prompt_tokens=[1], max_tokens=5) for i in range(5)]
    for s in seqs:
        sched.add(s)
    run_to_completion(sched)
    assert max(runner.batch_sizes) == 2
    for s in seqs:
        check(s, rec)


def test_continuous_admits_into_a_running_batch_but_static_waits():
    def run(scheduling):
        runner, rec = FakeRunner(), Recorder()
        sched = Scheduler(runner, make_backend("paged", runner), max_batch_size=8,
                          scheduling=scheduling, max_prefill_per_step=1,
                          on_token=rec.on_token, on_finish=rec.on_finish)
        sched.add(Sequence(id="a", prompt_tokens=[1], max_tokens=6))
        sched.add(Sequence(id="b", prompt_tokens=[1], max_tokens=6))
        run_to_completion(sched)
        return runner.batch_sizes

    assert 2 in run("continuous")        # b joins while a is running
    assert max(run("static")) == 1        # b waits until a is done


def test_prefill_per_step_bounds_admissions():
    runner, rec = FakeRunner(), Recorder()
    sched = Scheduler(runner, make_backend("paged", runner), max_batch_size=8,
                      max_prefill_per_step=2, on_token=rec.on_token, on_finish=rec.on_finish)
    for i in range(5):
        sched.add(Sequence(id=f"s{i}", prompt_tokens=[1], max_tokens=3))
    sched.step()
    assert runner.prefills == 2 and len(sched.running) == 2 and len(sched.waiting) == 3


@pytest.mark.parametrize("kind", ["paged", "padded"])
def test_preemption_when_memory_runs_out_still_completes_everyone(kind):
    runner, rec = FakeRunner(), Recorder()
    # Enough for roughly one long sequence at a time.
    budget = 16 if kind == "paged" else 64 * 2
    sched = Scheduler(runner, make_backend(kind, runner, tokens_budget=budget, block_size=4),
                      max_batch_size=8, on_token=rec.on_token, on_finish=rec.on_finish)
    seqs = [Sequence(id=f"s{i}", prompt_tokens=[1, 2, 3, 4, 5, 6, 7], max_tokens=20)
            for i in range(4)]
    for s in seqs:
        sched.add(s)
    run_to_completion(sched)
    for s in seqs:
        check(s, rec)
    if kind == "paged":
        assert sched.num_preemptions > 0
        assert sched.backend.bm.num_free == sched.backend.bm.num_blocks


def test_prompt_larger_than_budget_fails_immediately():
    runner, rec = FakeRunner(), Recorder()
    sched = Scheduler(runner, make_backend("paged", runner, tokens_budget=8, block_size=4),
                      on_token=rec.on_token, on_finish=rec.on_finish)
    seq = Sequence(id="big", prompt_tokens=list(range(20)))
    sched.add(seq)
    assert seq.status is SeqStatus.FINISHED
    assert seq.stop_reason == "error" and "exceeds" in seq.error
    assert rec.finished == ["big"]
    assert not sched.has_work()


def test_prefix_sharing_reduces_prefill_work():
    runner, rec = FakeRunner(), Recorder()
    backend = make_backend("paged", runner, block_size=4)
    sched = Scheduler(runner, backend, max_batch_size=8,
                      on_token=rec.on_token, on_finish=rec.on_finish)
    shared = list(range(1, 9))  # two full blocks
    a = Sequence(id="a", prompt_tokens=shared + [21], max_tokens=3)
    b = Sequence(id="b", prompt_tokens=shared + [31], max_tokens=3)
    sched.add(a)
    sched.add(b)
    run_to_completion(sched)
    check(a, rec)
    check(b, rec)
    assert backend.bm.cache_hits == 2
    assert backend.bm.num_free == backend.bm.num_blocks


def test_sequence_finishing_on_first_token_is_freed_before_decode():
    runner, rec = FakeRunner(), Recorder()
    backend = make_backend("paged", runner, block_size=4)
    sched = Scheduler(runner, backend, on_token=rec.on_token, on_finish=rec.on_finish)
    sched.add(Sequence(id="one", prompt_tokens=[1], max_tokens=1))
    sched.step()
    assert not sched.has_work()
    assert runner.decode_calls == 0
    assert backend.bm.num_free == backend.bm.num_blocks
