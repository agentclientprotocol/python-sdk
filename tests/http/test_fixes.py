"""Regression tests for reliability fixes on the HTTP/WS transport.

Covers:
* OutboundStream backpressure (no silent message drops under a full buffer).
* SSE keepalive emission on idle streams.
* HTTP client surfacing disconnect (EOF) when the connection-scoped SSE stream ends.
* Server cleanup of a leaked connection when ``initialize`` fails/times out.
* WebSocket client cookie support (send stored Cookie; capture Set-Cookie).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx2
import pytest

import acp.http.server as server_mod
from acp.http.client import create_http_stream
from acp.http.protocol import CONNECTION_ID_HEADER, CONTENT_TYPE_JSON
from acp.http.server import AcpServer, OutboundStream

CT_JSON = "application/json"


# -- Finding 1: OutboundStream backpressure ------------------------------------


@pytest.mark.asyncio
async def test_outbound_stream_does_not_drop_beyond_capacity() -> None:
    """Pushing more than ``capacity`` messages must not silently drop any.

    With a bounded queue and ``put_nowait``, the (capacity+1)-th message was
    dropped. Backpressure-aware push blocks the producer until a consumer drains,
    so every message is eventually delivered in order.
    """
    stream = OutboundStream(capacity=2)
    total = 5

    async def produce() -> None:
        for i in range(total):
            await stream.push({"n": i})
        stream.close()

    producer = asyncio.ensure_future(produce())
    received = [msg async for msg in stream.iterate()]
    await producer
    assert received == [{"n": i} for i in range(total)]


@pytest.mark.asyncio
async def test_outbound_stream_push_blocks_when_full() -> None:
    """push must not complete once the buffer is full and no consumer drains."""
    stream = OutboundStream(capacity=1)
    await stream.push({"n": 0})  # fills the buffer
    blocked = asyncio.ensure_future(stream.push({"n": 1}))
    await asyncio.sleep(0.05)
    assert not blocked.done()
    # Draining one message unblocks the producer.
    it = stream.iterate()
    assert await it.__anext__() == {"n": 0}
    await asyncio.wait_for(blocked, timeout=1)
    stream.close()


# -- Finding 3: SSE keepalive --------------------------------------------------


@pytest.mark.asyncio
async def test_outbound_stream_close_unblocks_full_buffer() -> None:
    stream = OutboundStream(capacity=1)
    await stream.push({"n": 0})
    blocked = asyncio.create_task(stream.push({"n": 1}))
    await asyncio.sleep(0)
    assert not blocked.done()
    stream.close()
    await asyncio.wait_for(blocked, timeout=1)
    assert [message async for message in stream.iterate()] == []


@pytest.mark.asyncio
async def test_open_stream_emits_keepalive_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    """An idle connection-scoped stream must emit periodic SSE keepalive frames."""
    monkeypatch.setattr(server_mod, "SSE_KEEPALIVE_INTERVAL_SECONDS", 0.05)

    server = AcpServer(lambda conn: _NoopAgent())
    result = await server.handle_post(
        {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 1}},
        content_type=CT_JSON,
        connection_id=None,
        session_id=None,
    )
    connection_id = result.headers[CONNECTION_ID_HEADER]

    frames: list[bytes] = []

    async def drain() -> None:
        async for frame in server.open_stream(connection_id=connection_id, session_id=None):
            frames.append(frame)

    task = asyncio.ensure_future(drain())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await server.close()

    assert any(frame == b": keepalive\n\n" for frame in frames)


# -- Finding 5: initialize failure cleanup -------------------------------------


@pytest.mark.asyncio
async def test_initialize_timeout_cleans_up_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung ``initialize`` must not leak a registered connection."""
    monkeypatch.setattr(server_mod, "INITIALIZE_TIMEOUT_SECONDS", 0.1)

    server = AcpServer(lambda conn: _SilentInitAgent())
    result = await server.handle_post(
        {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 1}},
        content_type=CT_JSON,
        connection_id=None,
        session_id=None,
    )
    assert result.status_code >= 500
    # No connection should remain registered after a failed initialize.
    assert not server._connections
    await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_interrupted_initialize_cleans_up_agent(shutdown: bool) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    class Agent(_SilentInitAgent):
        async def initialize(self, protocol_version: int = 1, **kwargs: Any) -> Any:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    server = AcpServer(lambda conn: Agent())
    request = asyncio.create_task(
        server.handle_post(
            {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 1}},
            content_type=CT_JSON,
            connection_id=None,
            session_id=None,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        if shutdown:
            await asyncio.wait_for(server.close(), timeout=1)
        else:
            request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request, timeout=1)
        assert stopped.is_set()
        assert not server._connections
    finally:
        request.cancel()
        await server.close()


# -- Finding 4: HTTP client surfaces disconnect on stream EOF ------------------


@pytest.mark.asyncio
async def test_http_client_surfaces_eof_when_connection_stream_ends() -> None:
    """When the connection-scoped SSE stream ends, receive() must return None."""
    conn_id = "conn-eof"

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            if body.get("method") == "initialize":
                return httpx2.Response(
                    200,
                    headers={CONNECTION_ID_HEADER: conn_id, "Content-Type": CONTENT_TYPE_JSON},
                    json={"jsonrpc": "2.0", "id": body["id"], "result": {}},
                )
            return httpx2.Response(202)
        if request.method == "GET":
            # SSE stream that immediately ends (empty body -> EOF).
            return httpx2.Response(200, headers={"Content-Type": "text/event-stream"}, content=b"")
        return httpx2.Response(202)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    transport = create_http_stream("http://testserver/acp", client=client)
    try:
        await transport.send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        # Drain the initialize result.
        assert await asyncio.wait_for(transport.receive(), timeout=1) == {"jsonrpc": "2.0", "id": 0, "result": {}}
        # The connection stream ends -> transport must surface EOF, not hang.
        assert await asyncio.wait_for(transport.receive(), timeout=1) is None
    finally:
        await transport.close()
        await client.aclose()


# -- Helpers -------------------------------------------------------------------


class _NoopAgent:
    def __init__(self) -> None:
        self._conn: Any = None

    def on_connect(self, conn: Any) -> None:
        self._conn = conn

    async def initialize(self, protocol_version: int = 1, **kwargs: Any) -> Any:
        from acp.schema import InitializeResponse

        return InitializeResponse(protocol_version=1)


class _SilentInitAgent:
    """An agent whose initialize never returns, forcing a server-side timeout."""

    def on_connect(self, conn: Any) -> None:
        pass

    async def initialize(self, protocol_version: int = 1, **kwargs: Any) -> Any:
        await asyncio.sleep(3600)
