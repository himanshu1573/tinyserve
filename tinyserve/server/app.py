"""OpenAI-compatible HTTP surface.

The handler never touches MLX. It creates a Sequence, hands it to the
engine thread, and awaits messages on an asyncio.Queue. This is a
correctness rule, not style: a blocking model call on the event loop
stalls every other connection, and the whole project is about serving
eight of them.

Endpoints:
    GET  /health                 liveness + the server's pid (the benchmark
                                 samples its RSS)
    GET  /stats                  scheduler, KV backend and MLX memory counters
    POST /stats/reset-peak       zero MLX's peak-memory counter (benchmark start)
    GET  /v1/models              what OpenAI clients call first
    POST /v1/completions         raw prompt, SSE or one JSON body
    POST /v1/chat/completions    messages -> chat template -> same engine
"""

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tinyserve.config import EngineConfig
from tinyserve.engine.engine import Engine
from tinyserve.engine.sequence import Sequence
from tinyserve.server.sse import (
    SSE_DONE,
    chat_chunk,
    chat_final,
    completion_chunk,
    completion_final,
    sse_frame,
    usage_for,
)


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = True
    model: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = True
    model: str | None = None


def create_app(runner, config: EngineConfig | None = None) -> FastAPI:
    config = config or EngineConfig()
    engine = Engine(runner, config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start()
        try:
            yield
        finally:
            engine.stop()

    app = FastAPI(title="tinyserve", lifespan=lifespan)
    app.state.engine = engine
    model_name = config.model

    def _submit(prompt_tokens: list[int], max_tokens: int, temperature: float,
                top_p: float, prefix: str) -> Sequence:
        seq = Sequence(
            id=f"{prefix}-{uuid.uuid4().hex[:12]}",
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            out_queue=asyncio.Queue(),
            loop=asyncio.get_running_loop(),
        )
        engine.submit(seq)
        return seq

    async def _collect(seq: Sequence) -> tuple[str, str]:
        pieces, reason = [], None
        while True:
            kind, value = await seq.out_queue.get()
            if kind == "text":
                pieces.append(value)
            elif kind == "error":
                reason = "error"
            else:
                return "".join(pieces), (reason or value)

    def _usage(seq: Sequence) -> dict:
        completion = sum(1 for t in seq.generated if t not in runner.eos_ids)
        return usage_for(len(seq.prompt_tokens), completion)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model_name, "pid": os.getpid()}

    @app.get("/stats")
    async def stats():
        return {"config": config.to_dict(), **engine.stats(), **runner.memory_stats()}

    @app.post("/stats/reset-peak")
    async def reset_peak():
        runner.reset_peak_memory()
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list",
                "data": [{"id": model_name, "object": "model", "owned_by": "tinyserve"}]}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        seq = _submit(runner.encode(req.prompt), req.max_tokens, req.temperature,
                      req.top_p, "cmpl")
        created = int(time.time())
        model = req.model or model_name

        if req.stream:
            async def gen() -> AsyncIterator[str]:
                while True:
                    kind, value = await seq.out_queue.get()
                    if kind == "text":
                        yield sse_frame(completion_chunk(seq.id, value, model, created))
                    elif kind == "error":
                        yield sse_frame({"error": {"message": value}})
                    else:
                        yield sse_frame(completion_final(seq.id, value, model, created,
                                                         _usage(seq)))
                        yield SSE_DONE
                        return
            return StreamingResponse(gen(), media_type="text/event-stream")

        text, reason = await _collect(seq)
        return {
            "id": seq.id, "object": "text_completion", "created": created,
            "model": model,
            "choices": [{"text": text, "index": 0, "logprobs": None,
                         "finish_reason": reason}],
            "usage": _usage(seq),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt = runner.format_chat([m.model_dump() for m in req.messages])
        seq = _submit(runner.encode(prompt), req.max_tokens, req.temperature,
                      req.top_p, "chatcmpl")
        created = int(time.time())
        model = req.model or model_name

        if req.stream:
            async def gen() -> AsyncIterator[str]:
                first = True
                while True:
                    kind, value = await seq.out_queue.get()
                    if kind == "text":
                        yield sse_frame(chat_chunk(seq.id, value, model, created,
                                                   role="assistant" if first else None))
                        first = False
                    elif kind == "error":
                        yield sse_frame({"error": {"message": value}})
                    else:
                        yield sse_frame(chat_final(seq.id, value, model, created,
                                                   _usage(seq)))
                        yield SSE_DONE
                        return
            return StreamingResponse(gen(), media_type="text/event-stream")

        text, reason = await _collect(seq)
        return {
            "id": seq.id, "object": "chat.completion", "created": created,
            "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": reason}],
            "usage": _usage(seq),
        }

    return app


def get_app() -> FastAPI:
    """Entry point for ``uvicorn tinyserve.server.app:get_app --factory``.
    Configuration comes from TINYSERVE_* environment variables so the
    factory signature stays uvicorn-shaped; ``tinyserve serve`` sets them."""
    from tinyserve.engine.runner import Runner

    config = EngineConfig(
        model=os.environ.get("TINYSERVE_MODEL", EngineConfig.model),
        max_batch_size=int(os.environ.get("TINYSERVE_MAX_BATCH", EngineConfig.max_batch_size)),
        scheduling=os.environ.get("TINYSERVE_SCHEDULING", EngineConfig.scheduling),
        kv_backend=os.environ.get("TINYSERVE_KV_BACKEND", EngineConfig.kv_backend),
        kv_budget_gb=float(os.environ.get("TINYSERVE_KV_GB", EngineConfig.kv_budget_gb)),
        block_size=int(os.environ.get("TINYSERVE_BLOCK_SIZE", EngineConfig.block_size)),
        prefix_caching=os.environ.get("TINYSERVE_PREFIX_CACHING", "1") == "1",
    ).validate()
    return create_app(Runner.load(config.model), config)
