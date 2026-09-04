"""Real model, real forward. Not mocked: a mocked forward pass tests nothing."""

import mlx.core as mx
import pytest

from tinyserve.engine.runner import Runner

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def runner():
    return Runner.load()


def test_prefill_returns_1d_logits(runner):
    ids = runner.encode("The capital of France is")
    logits = runner.prefill(ids, runner.new_cache())
    assert logits.ndim == 1 and logits.shape[0] > 1000


def test_decode_step_returns_1d_logits(runner):
    cache = runner.new_cache()
    first = runner.prefill(runner.encode("The capital of France is"), cache)
    logits = runner.decode_step(int(mx.argmax(first).item()), cache)
    assert logits.shape == first.shape


def test_greedy_continuation_is_coherent(runner):
    cache = runner.new_cache()
    logits = runner.prefill(runner.encode("The capital of France is"), cache)
    out = []
    for _ in range(4):
        tok = int(mx.argmax(logits).item())
        out.append(tok)
        logits = runner.decode_step(tok, cache)
    assert "Paris" in runner.tokenizer.decode(out)


def test_fresh_cache_reproduces_the_same_tokens(runner):
    def run():
        cache = runner.new_cache()
        logits = runner.prefill(runner.encode("Count: 1 2 3"), cache)
        out = []
        for _ in range(5):
            tok = int(mx.argmax(logits).item())
            out.append(tok)
            logits = runner.decode_step(tok, cache)
        return out

    assert run() == run()


def test_kv_spec_matches_qwen25_15b(runner):
    spec = runner.kv_spec
    assert spec.num_layers == 28 and spec.n_kv_heads == 2 and spec.head_dim == 128
    assert spec.bytes_per_token == 28 * 2 * 2 * 128 * spec.itemsize


def test_eos_ids_include_im_end(runner):
    assert runner.encode("<|im_end|>")[0] in runner.eos_ids
