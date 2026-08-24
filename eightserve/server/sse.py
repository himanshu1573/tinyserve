"""Server-sent-event framing and OpenAI-shaped completion payloads.

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


def completion_final(seq_id: str, finish_reason: str, model: str,
                     created: int) -> dict:
    return {
        "id": seq_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"text": "", "index": 0, "logprobs": None,
                     "finish_reason": finish_reason}],
    }
