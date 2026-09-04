import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from tinyserve.config import EngineConfig
from tinyserve.server.app import create_app
from tests.fakes import FakeRunner


@pytest.fixture
async def client():
    app = create_app(FakeRunner(), EngineConfig(kv_budget_gb=0.001, block_size=4))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


def frames(body: str):
    out = []
    for line in body.split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[len("data: "):]))
    return out


async def test_health_and_models(client):
    r = await client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and "pid" in r.json()
    r = await client.get("/v1/models")
    assert r.json()["data"][0]["object"] == "model"


async def test_stats_exposes_scheduler_and_backend(client):
    r = await client.get("/stats")
    body = r.json()
    assert body["backend"] == "paged" and body["running"] == 0
    assert body["config"]["kv_backend"] == "paged"
    assert "mlx_peak_gb" in body
    assert (await client.post("/stats/reset-peak")).json()["status"] == "ok"


async def test_completions_streams_sse_frames(client):
    async with client.stream("POST", "/v1/completions", json={
        "prompt": "hello", "max_tokens": 10, "stream": True,
    }) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in r.aiter_text()])
    assert body.endswith("data: [DONE]\n\n")
    fs = frames(body)
    assert "".join(f["choices"][0]["text"] for f in fs) == "<4><5><6><7><8><9>"
    assert fs[-1]["choices"][0]["finish_reason"] == "stop"
    assert fs[-1]["usage"]["completion_tokens"] == 6


async def test_completions_non_streaming_returns_one_json_body(client):
    r = await client.post("/v1/completions", json={
        "prompt": "hello", "max_tokens": 2, "stream": False,
    })
    payload = r.json()
    assert payload["choices"][0]["text"] == "<4><5>"
    assert payload["choices"][0]["finish_reason"] == "length"
    assert payload["usage"]["prompt_tokens"] == 3


async def test_chat_completions_stream_and_json(client):
    msgs = [{"role": "user", "content": "hi"}]
    async with client.stream("POST", "/v1/chat/completions", json={
        "messages": msgs, "max_tokens": 3, "stream": True,
    }) as r:
        body = "".join([chunk async for chunk in r.aiter_text()])
    fs = frames(body)
    assert fs[0]["object"] == "chat.completion.chunk"
    assert fs[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(f["choices"][0]["delta"].get("content", "") for f in fs) == "<4><5><6>"

    r = await client.post("/v1/chat/completions", json={
        "messages": msgs, "max_tokens": 3, "stream": False,
    })
    assert r.json()["choices"][0]["message"]["content"] == "<4><5><6>"


async def test_handler_does_not_block_the_event_loop(client):
    async def slow_request():
        r = await client.post("/v1/completions",
                              json={"prompt": "hello", "max_tokens": 10, "stream": False})
        return r.status_code

    task = asyncio.create_task(slow_request())
    health = await client.get("/health")
    assert health.status_code == 200
    assert await task == 200


async def test_concurrent_requests_all_complete(client):
    async def one(i):
        r = await client.post("/v1/completions",
                              json={"prompt": "x", "max_tokens": 4, "stream": False})
        return r.json()["choices"][0]["text"]

    texts = await asyncio.gather(*(one(i) for i in range(6)))
    assert texts == ["<4><5><6><7>"] * 6
