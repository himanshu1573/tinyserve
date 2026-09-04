import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from eightserve.server.app import create_app
from tests.test_engine import FakeRunner


@pytest.fixture
def client():
    app = create_app(FakeRunner())
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_health(client):
    async with client as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_completions_streams_sse_frames(client):
    async with client as c:
        async with c.stream("POST", "/v1/completions", json={
            "prompt": "hello", "max_tokens": 10, "stream": True,
        }) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = "".join([chunk async for chunk in r.aiter_text()])

    assert body.endswith("data: [DONE]\n\n")
    assert "<100>" in body and "<102>" in body


async def test_completions_non_streaming_returns_one_json_body(client):
    async with client as c:
        r = await c.post("/v1/completions", json={
            "prompt": "hello", "max_tokens": 10, "stream": False,
        })
    assert r.status_code == 200
    payload = r.json()
    assert payload["choices"][0]["text"] == "<100><101><102>"
    assert payload["choices"][0]["finish_reason"] == "stop"


async def test_handler_does_not_block_the_event_loop(client):
    """The rule the whole design rests on: while a completion is in
    flight, other requests still get served."""
    async with client as c:
        async def slow_request():
            r = await c.post("/v1/completions",
                             json={"prompt": "hello", "max_tokens": 10,
                                   "stream": False})
            return r.status_code

        task = asyncio.create_task(slow_request())
        health = await c.get("/health")
        assert health.status_code == 200
        assert await task == 200
