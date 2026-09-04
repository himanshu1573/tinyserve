import asyncio

import pytest

from eightserve.engine.engine import Engine
from eightserve.engine.sequence import Sequence, SeqStatus


class FakeRunner:
    """Emits tokens 100, 101, 102, then EOS. Deterministic, instant."""
    eos_id = 999

    def __init__(self):
        self.tokenizer = self
        self.detokenizer = self
        self._segments = []

    # --- runner surface ---
    def encode(self, text):
        return [1, 2, 3]

    def new_cache(self):
        return {"n": 0}

    def prefill(self, ids, cache):
        cache["n"] = 0
        return self._logits_for(100)

    def decode_step(self, token_id, cache):
        cache["n"] += 1
        nxt = [101, 102, 999][min(cache["n"] - 1, 2)]
        return self._logits_for(nxt)

    def _logits_for(self, token_id):
        import mlx.core as mx
        v = [0.0] * 1000
        v[token_id] = 10.0
        return mx.array(v)

    # --- detokenizer surface ---
    def reset(self):
        self._segments = []

    def add_token(self, t):
        self._segments.append(f"<{t}>")

    def finalize(self):
        pass

    @property
    def last_segment(self):
        # Mirrors mlx-lm's StreamingDetokenizer: reading the segment
        # consumes it, so a second read (after finalize) returns "".
        seg = "".join(self._segments)
        self._segments = []
        return seg


async def drain(seq):
    out, reason = [], None
    while True:
        kind, value = await asyncio.wait_for(seq.out_queue.get(), timeout=10)
        if kind == "text":
            out.append(value)
        elif kind == "error":
            raise AssertionError(f"engine error: {value}")
        else:
            reason = value
            break
    return "".join(out), reason


async def test_engine_streams_tokens_to_the_queue():
    engine = Engine(FakeRunner())
    engine.start()
    try:
        seq = Sequence(id="s1", prompt_tokens=[1, 2], max_tokens=10,
                       out_queue=asyncio.Queue(),
                       loop=asyncio.get_running_loop())
        engine.submit(seq)
        text, reason = await drain(seq)
        assert text == "<100><101><102>"
        assert reason == "stop"
        assert seq.status is SeqStatus.FINISHED
    finally:
        engine.stop()


async def test_max_tokens_truncates_and_reports_length():
    engine = Engine(FakeRunner())
    engine.start()
    try:
        seq = Sequence(id="s2", prompt_tokens=[1], max_tokens=2,
                       out_queue=asyncio.Queue(),
                       loop=asyncio.get_running_loop())
        engine.submit(seq)
        text, reason = await drain(seq)
        assert text == "<100><101>"
        assert reason == "length"
    finally:
        engine.stop()


async def test_two_sequences_do_not_interleave_or_share_cache():
    engine = Engine(FakeRunner())
    engine.start()
    try:
        loop = asyncio.get_running_loop()
        seqs = [Sequence(id=f"s{i}", prompt_tokens=[1], max_tokens=10,
                         out_queue=asyncio.Queue(), loop=loop)
                for i in range(2)]
        for s in seqs:
            engine.submit(s)
        results = await asyncio.gather(*(drain(s) for s in seqs))
        assert [r[0] for r in results] == ["<100><101><102>"] * 2
    finally:
        engine.stop()


async def test_engine_reports_errors_instead_of_hanging():
    class Broken(FakeRunner):
        def prefill(self, ids, cache):
            raise RuntimeError("boom")

    engine = Engine(Broken())
    engine.start()
    try:
        seq = Sequence(id="s3", prompt_tokens=[1], max_tokens=4,
                       out_queue=asyncio.Queue(),
                       loop=asyncio.get_running_loop())
        engine.submit(seq)
        kind, value = await asyncio.wait_for(seq.out_queue.get(), timeout=10)
        assert kind == "error"
        assert "boom" in value
        kind, _ = await asyncio.wait_for(seq.out_queue.get(), timeout=10)
        assert kind == "done"
    finally:
        engine.stop()
