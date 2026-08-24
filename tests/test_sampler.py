import mlx.core as mx
import pytest

from eightserve.engine.sampler import sample


def test_greedy_picks_argmax():
    logits = mx.array([1.0, 5.0, 3.0, 2.0])
    assert sample(logits, temperature=0.0) == 1


def test_greedy_is_deterministic():
    logits = mx.array([0.1, 0.2, 9.0, 0.3])
    assert [sample(logits, temperature=0.0) for _ in range(5)] == [2] * 5


def test_temperature_zero_ignores_top_p():
    logits = mx.array([1.0, 5.0, 3.0])
    assert sample(logits, temperature=0.0, top_p=0.01) == 1


def test_top_p_excludes_low_probability_tokens():
    # Token 0 holds almost all the mass; nucleus at 0.5 can only contain it.
    logits = mx.array([20.0, 0.0, 0.0, 0.0])
    picks = {sample(logits, temperature=1.0, top_p=0.5) for _ in range(50)}
    assert picks == {0}


def test_top_p_one_can_reach_any_token():
    # A flat distribution with the full nucleus should eventually pick
    # something other than index 0.
    logits = mx.zeros((8,))
    picks = {sample(logits, temperature=1.0, top_p=1.0) for _ in range(200)}
    assert len(picks) > 1


def test_rejects_non_1d_logits():
    with pytest.raises(ValueError):
        sample(mx.zeros((1, 4)), temperature=0.0)
