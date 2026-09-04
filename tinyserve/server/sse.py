"""Server-sent-event framing and OpenAI-shaped payloads.

Pure string and dict construction, so the wire format is testable without
a server, a client, or a model.
"""

import json

SSE_DONE = "data: [DONE]\n\n"


def sse_frame(payload: dict) -> str:
    """One SSE event. The blank line is the terminator — without it the
    client buffers forever waiting for the event to end."""
    return f"data: {json.dumps(payload)}\n\n"


def completion_chunk(seq_id: str, text: str, model: str, created: int) -> dict:
    return {
        "id": seq_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": None,
                     "finish_reason": None}],
    }


def completion_final(seq_id: str, finish_reason: str, model: str, created: int,
                     usage: dict | None = None) -> dict:
    payload = {
        "id": seq_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"text": "", "index": 0, "logprobs": None,
                     "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def chat_chunk(seq_id: str, text: str, model: str, created: int,
               role: str | None = None) -> dict:
    delta = {"content": text}
    if role:
        delta["role"] = role
    return {
        "id": seq_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


def chat_final(seq_id: str, finish_reason: str, model: str, created: int,
               usage: dict | None = None) -> dict:
    payload = {
        "id": seq_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def usage_for(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
