import asyncio

from tinyserve.config import EngineConfig
from tinyserve.engine.engine import Engine
from tinyserve.engine.sequence import Sequence, SeqStatus
from tests.fakes import BrokenRunner, FakeRunner


def small_config(**kw):
    defaults = dict(kv_budget_gb=0.001, block_size=4)
    return EngineConfig(**{**defaults, **kw})


async def drain(seq):
    out, reason, error = [], None, None
    while True:
        kind, value = await asyncio.wait_for(seq.out_queue.get(), timeout=10)
        if kind == "text":
            out.append(value)
        elif kind == "error":
            error = value
        else:
            reason = value
            break
    return "".join(out), reason, error


def new_seq(id, prompt, max_tokens):
    return Sequence(id=id, prompt_tokens=prompt, max_tokens=max_tokens,
                    out_queue=asyncio.Queue(), loop=asyncio.get_running_loop())


async def test_engine_streams_tokens_to_the_queue():
    engine = Engine(FakeRunner(), small_config())
    engine.start()
    try:
        seq = new_seq("s1", [1, 2, 3], 10)
        engine.submit(seq)
        text, reason, error = await drain(seq)
        assert text == "<4><5><6><7><8><9>"
        assert reason == "stop" and error is None
        assert seq.status is SeqStatus.FINISHED
        assert seq.t_first_token > seq.t_submitted > 0
    finally:
        engine.stop()


async def test_max_tokens_truncates_and_reports_length():
    engine = Engine(FakeRunner(), small_config())
    engine.start()
    try:
        seq = new_seq("s2", [1], 2)
        engine.submit(seq)
        text, reason, _ = await drain(seq)
        assert text == "<2><3>" and reason == "length"
    finally:
        engine.stop()


async def test_concurrent_sequences_get_their_own_continuations():
    engine = Engine(FakeRunner(), small_config(max_batch_size=4))
    engine.start()
    try:
        seqs = [new_seq(f"s{i}", [i], 10) for i in range(1, 5)]
        for s in seqs:
            engine.submit(s)
        results = await asyncio.gather(*(drain(s) for s in seqs))
        for i, (text, reason, _) in zip(range(1, 5), results):
            assert text == "".join(f"<{t}>" for t in range(i + 1, 10))
            assert reason == "stop"
        assert engine.stats()["running"] == 0
    finally:
        engine.stop()


async def test_engine_reports_errors_instead_of_hanging():
    engine = Engine(BrokenRunner(), small_config())
    engine.start()
    try:
        seq = new_seq("s3", [1], 4)
        engine.submit(seq)
        text, reason, error = await drain(seq)
        assert reason == "error" and "boom" in error
    finally:
        engine.stop()


async def test_serial_config_reproduces_batch_size_one():
    runner = FakeRunner()
    engine = Engine(runner, small_config(max_batch_size=1))
    engine.start()
    try:
        seqs = [new_seq(f"s{i}", [1], 5) for i in range(3)]
        for s in seqs:
            engine.submit(s)
        await asyncio.gather(*(drain(s) for s in seqs))
        assert max(runner.batch_sizes) == 1
    finally:
        engine.stop()
