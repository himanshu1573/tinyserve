import mlx.core as mx
import pytest

from tinyserve.engine.sampler import sample, sample_batch
from tinyserve.engine.sequence import Sequence


def test_greedy_picks_argmax():
    assert sample(mx.array([1.0, 5.0, 3.0, 2.0]), temperature=0.0) == 1


def test_greedy_is_deterministic():
    logits = mx.array([0.1, 0.2, 9.0, 0.3])
    assert [sample(logits, temperature=0.0) for _ in range(5)] == [2] * 5


def test_temperature_zero_ignores_top_p():
    assert sample(mx.array([1.0, 5.0, 3.0]), temperature=0.0, top_p=0.01) == 1


def test_top_p_excludes_low_probability_tokens():
    logits = mx.array([20.0, 0.0, 0.0, 0.0])
    picks = {sample(logits, temperature=1.0, top_p=0.5) for _ in range(50)}
    assert picks == {0}


def test_top_p_one_can_reach_any_token():
    picks = {sample(mx.zeros((8,)), temperature=1.0, top_p=1.0) for _ in range(200)}
    assert len(picks) > 1


def test_rejects_non_1d_logits():
    with pytest.raises(ValueError):
        sample(mx.zeros((1, 4)), temperature=0.0)


def test_sample_batch_greedy_rows_are_independent():
    logits = mx.array([[0.0, 9.0, 0.0], [9.0, 0.0, 0.0], [0.0, 0.0, 9.0]])
    seqs = [Sequence(id=str(i), prompt_tokens=[1]) for i in range(3)]
    assert sample_batch(logits, seqs) == [1, 0, 2]


def test_sample_batch_mixed_temperatures():
    logits = mx.array([[0.0, 9.0, 0.0], [30.0, 0.0, 0.0]])
    seqs = [Sequence(id="a", prompt_tokens=[1]),
            Sequence(id="b", prompt_tokens=[1], temperature=1.0)]
    assert sample_batch(logits, seqs) == [1, 0]


def test_sample_batch_rejects_row_mismatch():
    with pytest.raises(ValueError):
        sample_batch(mx.zeros((2, 3)), [Sequence(id="a", prompt_tokens=[1])])
