import json

from tinyserve.server.sse import (
    SSE_DONE, chat_chunk, chat_final, completion_chunk, completion_final,
    sse_frame, usage_for,
)


def test_frame_is_data_line_with_blank_line_terminator():
    assert sse_frame({"a": 1}) == 'data: {"a": 1}\n\n'


def test_frame_has_no_raw_newlines_in_payload():
    frame = sse_frame({"text": "line one\nline two"})
    assert frame.count("\n") == 2 and frame.endswith("\n\n")


def test_done_sentinel_is_literal():
    assert SSE_DONE == "data: [DONE]\n\n"


def test_completion_chunk_shape():
    c = completion_chunk("s1", "hello", model="m", created=1)
    assert c["object"] == "text_completion"
    assert c["choices"][0]["text"] == "hello"
    assert c["choices"][0]["finish_reason"] is None


def test_completion_final_carries_finish_reason_and_usage():
    c = completion_final("s1", "length", model="m", created=1, usage=usage_for(3, 5))
    assert c["choices"][0]["finish_reason"] == "length"
    assert c["choices"][0]["text"] == ""
    assert c["usage"] == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


def test_chat_chunks():
    c = chat_chunk("s1", "hi", model="m", created=1, role="assistant")
    assert c["object"] == "chat.completion.chunk"
    assert c["choices"][0]["delta"] == {"content": "hi", "role": "assistant"}
    f = chat_final("s1", "stop", model="m", created=1)
    assert f["choices"][0]["delta"] == {} and f["choices"][0]["finish_reason"] == "stop"


def test_chunk_round_trips_through_a_frame():
    frame = sse_frame(completion_chunk("s1", "hi", model="m", created=1))
    payload = json.loads(frame[len("data: "):].strip())
    assert payload["choices"][0]["text"] == "hi"
