"""HTTP boundaries, mounting, and streaming cleanup of the Starlette app."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Mount
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from acp.http.asgi import create_asgi_app
from acp.http.protocol import CONNECTION_ID_HEADER
from tests.conftest import TestAgent

INITIALIZE = {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 1}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "headers", "body", "status"),
    [
        ("POST", {"Content-Type": "text/plain"}, "{}", 415),
        ("POST", {"Content-Type": "application/json"}, "invalid", 400),
        ("POST", {"Content-Type": "application/json"}, "null", 400),
        ("POST", {"Content-Type": "application/json"}, "[]", 501),
        ("GET", {"Accept": "application/json"}, "", 406),
        ("GET", {"Accept": "text/event-stream"}, "", 400),
        ("GET", {"Accept": "text/event-stream", CONNECTION_ID_HEADER: "unknown"}, "", 404),
        ("DELETE", {}, "", 400),
        ("DELETE", {CONNECTION_ID_HEADER: "unknown"}, "", 404),
    ],
)
async def test_http_errors(method: str, headers: dict[str, str], body: str, status: int) -> None:
    app = create_asgi_app(lambda conn: TestAgent())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.request(method, "/acp", headers=headers, content=body)
        assert response.status_code == status
        assert "error" in response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["PUT", "OPTIONS"])
async def test_unsupported_methods_do_not_open_a_stream(method: str) -> None:
    app = create_asgi_app(lambda conn: TestAgent())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.request(method, "/acp")
        assert response.status_code == 405
        assert "allow" in response.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("endpoint", [None, "/rpc"])
async def test_direct_and_mounted_app_support_http_and_websocket(
    mounted: bool, endpoint: str | None, serve_asgi
) -> None:
    acp_app = (
        create_asgi_app(lambda conn: TestAgent())
        if endpoint is None
        else create_asgi_app(lambda conn: TestAgent(), path=endpoint)
    )
    assert isinstance(acp_app, Starlette)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        async with acp_app.router.lifespan_context(acp_app):
            yield

    app = Starlette(routes=[Mount("/agents", app=acp_app)], lifespan=lifespan) if mounted else acp_app
    prefix = "/agents" if mounted else ""
    path = prefix + (endpoint or "/acp")
    server = await serve_asgi(app)
    async with httpx.AsyncClient(base_url=f"http://{server.host}:{server.port}") as client:
        response = await client.post(path, json=INITIALIZE)
        assert response.status_code == 200
        assert response.json()["id"] == 0
        connection_id = response.headers[CONNECTION_ID_HEADER]
        deleted = await client.delete(path, headers={CONNECTION_ID_HEADER: connection_id})
        assert deleted.status_code == 202
        assert deleted.content == b""
        assert (await client.delete(path, headers={CONNECTION_ID_HEADER: connection_id})).status_code == 404

        async with connect(f"ws://{server.host}:{server.port}{path}") as websocket:
            assert websocket.response is not None
            assert CONNECTION_ID_HEADER in websocket.response.headers
            await websocket.send(json.dumps(INITIALIZE))
            response_body = await asyncio.wait_for(websocket.recv(), timeout=1)
            assert json.loads(response_body)["result"]["protocolVersion"] == 1

        wrong_path = prefix + ("/acp" if endpoint else "/other")
        for method in ("POST", "GET", "DELETE"):
            assert (await client.request(method, wrong_path, json=INITIALIZE)).status_code == 404
        with pytest.raises(InvalidStatus) as exc:
            async with connect(f"ws://{server.host}:{server.port}{wrong_path}"):
                pytest.fail("WebSocket connected outside the configured path")
        assert exc.value.response.status_code == 403


@pytest.mark.asyncio
async def test_lifespan_closes_http_connections() -> None:
    app = create_asgi_app(lambda conn: TestAgent())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            response = await client.post("/acp", json=INITIALIZE)
            connection_id = response.headers[CONNECTION_ID_HEADER]
        response = await client.delete("/acp", headers={CONNECTION_ID_HEADER: connection_id})
        assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/acp", "/rpc"])
async def test_sse_disconnect_releases_reader_before_reopening(path: str) -> None:
    connections = []

    def factory(conn):
        connections.append(conn)
        return TestAgent()

    app = create_asgi_app(factory, path=path)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(path, json=INITIALIZE)
        connection_id = response.headers[CONNECTION_ID_HEADER]
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "method": "GET",
            "path": path,
            "root_path": "",
            "query_string": b"",
            "headers": [
                (b"accept", b"text/event-stream"),
                (CONNECTION_ID_HEADER.lower().encode(), connection_id.encode()),
            ],
        }
        disconnect = asyncio.Event()
        started = asyncio.Event()

        async def receive() -> dict[str, Any]:
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                assert message["status"] == 200
                started.set()

        first = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            disconnect.set()
            await asyncio.wait_for(first, timeout=1)
        finally:
            first.cancel()
        # A stale reader must not consume output intended for the reopened GET.
        await connections[0].ext_notification("test", {"message": "after disconnect"})
        disconnect.clear()
        frames = []

        async def collect(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.body" and message.get("body"):
                frames.append(message["body"])
                disconnect.set()

        await asyncio.wait_for(app(scope, receive, collect), timeout=1)
        assert json.loads(frames[0].removeprefix(b"data: "))["params"] == {"message": "after disconnect"}
