import mlx.core as mx
import pytest

from eightserve.engine.runner import Runner

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def runner():
    return Runner.load()


def test_prefill_returns_1d_logits(runner):
    ids = runner.encode("The capital of France is")
    logits = runner.prefill(ids, runner.new_cache())
    assert logits.ndim == 1
    assert logits.shape[0] > 1000  # vocab, not sequence length


def test_decode_step_returns_1d_logits(runner):
    cache = runner.new_cache()
    ids = runner.encode("The capital of France is")
    first = runner.prefill(ids, cache)
    nxt = int(mx.argmax(first).item())
    logits = runner.decode_step(nxt, cache)
    assert logits.shape == first.shape


def test_greedy_continuation_is_coherent(runner):
    """The real end-to-end check: our own loop, not mlx-lm's."""
    cache = runner.new_cache()
    ids = runner.encode("The capital of France is")
    logits = runner.prefill(ids, cache)
    out = []
    for _ in range(4):
        tok = int(mx.argmax(logits).item())
        out.append(tok)
        logits = runner.decode_step(tok, cache)
    assert "Paris" in runner.tokenizer.decode(out)


def test_fresh_cache_reproduces_the_same_tokens(runner):
    """Two independent caches, same prompt, greedy -> identical output.
    Catches cache state leaking between sequences, which is exactly the
    bug that would silently corrupt batched decoding in Session 4."""
    def run():
        cache = runner.new_cache()
        ids = runner.encode("Count: 1 2 3")
        logits = runner.prefill(ids, cache)
        out = []
        for _ in range(5):
            tok = int(mx.argmax(logits).item())
            out.append(tok)
            logits = runner.decode_step(tok, cache)
        return out

    assert run() == run()
