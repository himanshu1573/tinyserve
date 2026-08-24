from eightserve.engine.sequence import Sequence, SeqStatus


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


def test_all_tokens_is_prompt_plus_generated():
    seq = make_seq()
    seq.append(10)
    assert seq.all_tokens() == [1, 2, 3, 10]


def test_stops_at_max_tokens():
    seq = make_seq(max_tokens=2)
    seq.append(10)
    assert not seq.should_stop(eos_id=999)
    seq.append(11)
    assert seq.should_stop(eos_id=999)
    assert seq.stop_reason == "length"


def test_stops_on_eos():
    seq = make_seq(max_tokens=100)
    seq.append(999)
    assert seq.should_stop(eos_id=999)
    assert seq.stop_reason == "stop"


def test_eos_beats_length_when_both_apply():
    seq = make_seq(max_tokens=1)
    seq.append(999)
    assert seq.should_stop(eos_id=999)
    assert seq.stop_reason == "stop"
