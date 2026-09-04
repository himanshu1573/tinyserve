from tinyserve.engine.sequence import Sequence, SeqStatus


def make_seq(**kw):
    defaults = dict(id="s1", prompt_tokens=[1, 2, 3], max_tokens=4)
    return Sequence(**{**defaults, **kw})


def test_starts_waiting():
    assert make_seq().status is SeqStatus.WAITING


def test_append_accumulates_generated_tokens():
    seq = make_seq()
    seq.append(10)
    seq.append(11)
    assert seq.generated == [10, 11]
    assert seq.all_tokens() == [1, 2, 3, 10, 11]
    assert seq.num_tokens == 5
    assert seq.last_token == 11


def test_last_token_before_generation_is_prompt_end():
    assert make_seq().last_token == 3


def test_stops_at_max_tokens():
    seq = make_seq(max_tokens=2)
    seq.append(10)
    assert not seq.should_stop(999)
    seq.append(11)
    assert seq.should_stop(999)
    assert seq.stop_reason == "length"


def test_stops_on_any_eos_id():
    seq = make_seq(max_tokens=100)
    seq.append(998)
    assert seq.should_stop({998, 999})
    assert seq.stop_reason == "stop"


def test_eos_beats_length_when_both_apply():
    seq = make_seq(max_tokens=1)
    seq.append(999)
    assert seq.should_stop(999)
    assert seq.stop_reason == "stop"


def test_block_helpers():
    seq = make_seq(prompt_tokens=list(range(10)))
    assert seq.num_blocks(4) == 3
    assert seq.block_tokens(0, 4) == [0, 1, 2, 3]
    assert seq.block_tokens(2, 4) == [8, 9]


def test_reset_kv_state_keeps_tokens():
    seq = make_seq()
    seq.append(7)
    seq.block_table = [3, 4]
    seq.num_cached_tokens = 4
    seq.reset_kv_state()
    assert seq.block_table == [] and seq.num_cached_tokens == 0
    assert seq.generated == [7]
