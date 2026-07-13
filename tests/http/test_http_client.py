"""Unit tests for the Streamable HTTP client transport (ported from http-stream.test.ts)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from acp._sse import serialize_sse_event
from acp.http.client import AcpHttpStatusError, create_http_stream
from acp.http.protocol import CONNECTION_ID_HEADER, CONTENT_TYPE_JSON, SESSION_ID_HEADER

CONN_ID = "conn-123"


class FakeServer:
    """A minimal in-memory Streamable HTTP server backed by httpx.MockTransport."""

    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []
        self.deleted = False
        # Queues feeding the connection-scoped and session-scoped SSE streams.
        self.conn_stream: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.session_streams: dict[str, asyncio.Queue[bytes | None]] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return self._handle_post(request)
        if request.method == "GET":
            return self._handle_get(request)
        if request.method == "DELETE":
            self.deleted = True
            return httpx.Response(202)
        return httpx.Response(405)

    def _handle_post(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.posts.append(body)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                headers={CONNECTION_ID_HEADER: CONN_ID, "Content-Type": CONTENT_TYPE_JSON},
                json={"jsonrpc": "2.0", "id": body["id"], "result": {"protocolVersion": 1}},
            )
        return httpx.Response(202)

    def _handle_get(self, request: httpx.Request) -> httpx.Response:
        session_id = request.headers.get(SESSION_ID_HEADER)
        if session_id is not None:
            queue = self.session_streams.setdefault(session_id, asyncio.Queue())
        else:
            queue = self.conn_stream

        async def body() -> Any:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    return
                yield chunk

        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=_AsyncByteStream(body()))

    def push_conn(self, message: dict[str, Any]) -> None:
        self.conn_stream.put_nowait(serialize_sse_event(message))

    def push_session(self, session_id: str, message: dict[str, Any]) -> None:
        queue = self.session_streams.setdefault(session_id, asyncio.Queue())
        queue.put_nowait(serialize_sse_event(message))


class _AsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, iterator: Any) -> None:
        self._iterator = iterator

    async def __aiter__(self) -> Any:
        async for chunk in self._iterator:
            yield chunk


def _make_transport(server: FakeServer):
    client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    return create_http_stream("http://testserver/acp", client=client), client


@pytest.mark.asyncio
async def test_initialize_posts_and_reads_connection_id() -> None:
    server = FakeServer()
    transport, client = _make_transport(server)
    try:
        await transport.send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        # The initialize result is enqueued back for the core to correlate.
        result = await asyncio.wait_for(transport.receive(), timeout=1)
        assert result == {"jsonrpc": "2.0", "id": 0, "result": {"protocolVersion": 1}}
        assert server.posts[0]["method"] == "initialize"
    finally:
        await transport.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_initialize_failure_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = create_http_stream("http://testserver/acp", client=client)
    try:
        with pytest.raises(AcpHttpStatusError) as exc:
            await transport.send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        assert exc.value.status_code == 500  # type: ignore[attr-defined]
    finally:
        await transport.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_connection_scoped_sse_delivers_new_session_result() -> None:
    server = FakeServer()
    transport, client = _make_transport(server)
    try:
        await transport.send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        await asyncio.wait_for(transport.receive(), timeout=1)  # drain initialize result
        # session/new POST returns 202; the result comes over the connection stream.
        await transport.send({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}})
        await asyncio.sleep(0.05)
        server.push_conn({"jsonrpc": "2.0", "id": 1, "result": {"sessionId": "sess-1"}})
        msg = await asyncio.wait_for(transport.receive(), timeout=1)
        assert msg == {"jsonrpc": "2.0", "id": 1, "result": {"sessionId": "sess-1"}}
        assert any(p.get("method") == "session/new" for p in server.posts)
    finally:
        await transport.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_session_scoped_sse_opens_after_new_session() -> None:
    server = FakeServer()
    transport, client = _make_transport(server)
    try:
        await transport.send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        await asyncio.wait_for(transport.receive(), timeout=1)
        await transport.send({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}})
        await asyncio.sleep(0.05)
        server.push_conn({"jsonrpc": "2.0", "id": 1, "result": {"sessionId": "sess-1"}})
        await asyncio.wait_for(transport.receive(), timeout=1)  # session/new result
        # Give the client time to open the session-scoped stream.
        await asyncio.sleep(0.05)
        assert "sess-1" in server.session_streams
        # A session-scoped notification arrives on the merged feed.
        server.push_session("sess-1", {"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "sess-1"}})
        msg = await asyncio.wait_for(transport.receive(), timeout=1)
        assert msg["method"] == "session/update"
    finally:
        await transport.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_session_scoped_post_sends_session_header() -> None:
    server = FakeServer()
    transport, client = _make_transport(server)
    try:
        await transport.send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        await asyncio.wait_for(transport.receive(), timeout=1)
        await transport.send({"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {"sessionId": "sess-1"}})
        prompt_post = next(p for p in server.posts if p.get("method") == "session/prompt")
        assert prompt_post["params"]["sessionId"] == "sess-1"
    finally:
        await transport.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_close_deletes_connection() -> None:
    server = FakeServer()
    transport, client = _make_transport(server)
    await transport.send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
    await asyncio.wait_for(transport.receive(), timeout=1)
    await transport.close()
    assert server.deleted is True
    # After close, receive() yields EOF.
    assert await asyncio.wait_for(transport.receive(), timeout=1) is None
    await client.aclose()


@pytest.mark.asyncio
async def test_post_error_status_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            if body.get("method") == "initialize":
                return httpx.Response(
                    200,
                    headers={CONNECTION_ID_HEADER: CONN_ID},
                    json={"jsonrpc": "2.0", "id": body["id"], "result": {}},
                )
            return httpx.Response(404)
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = create_http_stream("http://testserver/acp", client=client)
    try:
        await transport.send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        await asyncio.wait_for(transport.receive(), timeout=1)
        with pytest.raises(AcpHttpStatusError) as exc:
            await transport.send({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}})
        assert exc.value.status_code == 404  # type: ignore[attr-defined]
    finally:
        await transport.close()
        await client.aclose()
