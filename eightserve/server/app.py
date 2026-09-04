"""OpenAI-compatible HTTP surface.

The handler never touches MLX. It creates a Sequence, hands it to the
engine thread, and awaits tokens on an asyncio.Queue. Compatibility with
/v1/completions is deliberate: any existing client can point at this.
"""

import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from eightserve.engine.engine import Engine
from eightserve.engine.runner import Runner, DEFAULT_MODEL
from eightserve.engine.sequence import Sequence
from eightserve.server.sse import (
    SSE_DONE,
    completion_chunk,
    completion_final,
    sse_frame,
)


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = True
    model: str = DEFAULT_MODEL


def create_app(runner) -> FastAPI:
    app = FastAPI(title="eightserve")
    engine = Engine(runner)

    @app.on_event("startup")
    def _start():
        engine.start()

    @app.on_event("shutdown")
    def _stop():
        engine.stop()

    def _new_sequence(req: CompletionRequest) -> Sequence:
        import asyncio
        return Sequence(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            prompt_tokens=runner.tokenizer.encode(req.prompt),
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            out_queue=asyncio.Queue(),
            loop=asyncio.get_running_loop(),
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": DEFAULT_MODEL}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        seq = _new_sequence(req)
        engine.submit(seq)
        created = int(time.time())

        if req.stream:
            return StreamingResponse(
                _stream(seq, req.model, created),
                media_type="text/event-stream",
            )

        pieces, reason = [], "stop"
        while True:
            kind, value = await seq.out_queue.get()
            if kind == "text":
                pieces.append(value)
            elif kind == "error":
                reason = "error"
            else:
                reason = value if reason != "error" else "error"
                break

        return {
            "id": seq.id,
            "object": "text_completion",
            "created": created,
            "model": req.model,
            "choices": [{"text": "".join(pieces), "index": 0,
                         "logprobs": None, "finish_reason": reason}],
        }

    return app


async def _stream(seq: Sequence, model: str, created: int) -> AsyncIterator[str]:
    while True:
        kind, value = await seq.out_queue.get()
        if kind == "text":
            yield sse_frame(completion_chunk(seq.id, value, model, created))
        elif kind == "error":
            yield sse_frame(completion_final(seq.id, "error", model, created))
        else:
            yield sse_frame(completion_final(seq.id, value, model, created))
            yield SSE_DONE
            return


app = None


def get_app() -> FastAPI:
    """Entry point for `uvicorn eightserve.server.app:get_app --factory`.
    Loading the model here rather than at import keeps the test suite from
    pulling 1 GB off disk."""
    return create_app(Runner.load())
