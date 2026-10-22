from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx2
import pytest
import pytest_asyncio
from websockets.asyncio.server import serve

from acp.connection import Connection
from tests.http.test_http_client import FakeServer


@dataclass
class _Peer:
    connection: Connection
    send: Callable[[str], Awaitable[None]]
    receive: Callable[[], Awaitable[dict[str, Any]]]


async def _handler(method: str, params: Any, is_notification: bool) -> None:
    pass


@pytest_asyncio.fixture(params=["stdio", "websocket", "sse"])
async def peer(request: pytest.FixtureRequest, server) -> AsyncIterator[_Peer]:
    if request.param == "stdio":

        async def send(text: str) -> None:
            server.server_writer.write((text + "\n").encode())
            await server.server_writer.drain()

        async def receive() -> dict[str, Any]:
            return json.loads(await server.server_reader.readline())

        async with Connection(_handler, server.client_writer, server.client_reader) as connection:
            yield _Peer(connection, send, receive)
    elif request.param == "websocket":
        from acp.ws.client import create_websocket_stream

        socket_ready = asyncio.get_running_loop().create_future()

        async def accept(socket: Any) -> None:
            socket_ready.set_result(socket)
            await socket.wait_closed()

        async with serve(accept, "127.0.0.1", 0) as listener:
            transport = await create_websocket_stream(f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}")
            socket = await socket_ready

            async def receive() -> dict[str, Any]:
                return json.loads(await socket.recv())

            async with Connection(_handler, transport) as connection:
                yield _Peer(connection, socket.send, receive)
    else:
        fake = FakeServer()
        posts: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def handle(request: httpx2.Request) -> httpx2.Response:
            if request.method == "POST":
                posts.put_nowait(json.loads(request.content))
            return fake.handler(request)

        from acp.http.client import create_http_stream

        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
            transport = create_http_stream("http://test/acp", client=client)
            await transport.send({"jsonrpc": "2.0", "id": "initialize", "method": "initialize", "params": {}})
            await transport.receive()
            await posts.get()

            async def send(text: str) -> None:
                fake.conn_stream.put_nowait(("data: " + text + "\n\n").encode())

            async with Connection(_handler, transport) as connection:
                yield _Peer(connection, send, posts.get)


async def _request(peer: _Peer) -> tuple[asyncio.Task[Any], dict[str, Any]]:
    pending = asyncio.create_task(peer.connection.send_request("probe", {}))
    request = await asyncio.wait_for(peer.receive(), timeout=1)
    assert request["method"] == "probe"
    return pending, request


async def _complete(peer: _Peer, pending: asyncio.Task[Any], request: dict[str, Any]) -> None:
    values = [None, [], {}, False, 0, "", {"nested": ["雪🙂"]}]
    await peer.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": values}))
    assert await asyncio.wait_for(pending, timeout=1) == values


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ("invalid json", -32700),
        ("null", -32600),
        ("[]", -32600),
        ("{}", -32600),
        ("42", -32600),
        ("1.5", -32600),
        ('"text"', -32600),
        ("true", -32600),
        ("false", -32600),
    ],
)
async def test_invalid_payload_reports_error_and_preserves_requests(peer: _Peer, payload: str, code: int) -> None:
    pending, request = await _request(peer)
    await peer.send(payload)
    error = await asyncio.wait_for(peer.receive(), timeout=1)
    assert error["jsonrpc"] == "2.0" and error["id"] is None
    assert error["error"]["code"] == code
    assert not pending.done()
    await _complete(peer, pending, request)
    next_pending, next_request = await _request(peer)
    await _complete(peer, next_pending, next_request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload", ['{"jsonrpc":"2.0","id":[],"result":null}', '[{"jsonrpc":"2.0","id":123,"result":null}]']
)
async def test_ignored_payload_preserves_requests_without_reply(peer: _Peer, payload: str) -> None:
    pending, request = await _request(peer)
    await peer.send(payload)
    await _complete(peer, pending, request)
    next_pending, next_request = await _request(peer)
    await _complete(peer, next_pending, next_request)
