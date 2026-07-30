"""WebSocket transport tests (client + ASGI server handler)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from websockets.asyncio.server import serve

from acp.http.protocol import CONNECTION_ID_HEADER
from acp.http.server import AcpServer
from acp.schema import NewSessionResponse, PromptResponse
from acp.ws.client import create_websocket_stream
from acp.ws.server import handle_asgi_websocket
from tests.conftest import TestAgent


class _Agent(TestAgent):
    def __init__(self) -> None:
        super().__init__()
        self._conn: Any = None

    def on_connect(self, conn: Any) -> None:
        self._conn = conn

    async def new_session(self, cwd: str | None = None, mcp_servers: Any = None, **kwargs: Any) -> NewSessionResponse:
        return NewSessionResponse(session_id="sess-ws")

    async def prompt(self, session_id: str, prompt: Any = None, **kwargs: Any) -> PromptResponse:
        await self._conn.session_update(
            session_id=session_id,
            update={"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "yo"}},
        )
        return PromptResponse(stop_reason="end_turn")


# -- Client transport against a plain echo websocket server --------------------


@pytest.mark.asyncio
async def test_client_transport_send_receive_text_frames() -> None:
    async def echo(ws: Any) -> None:
        async for msg in ws:
            data = json.loads(msg)
            await ws.send(json.dumps({"echo": data}))

    async with serve(echo, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = await create_websocket_stream(f"ws://localhost:{port}")
        try:
            await transport.send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
            received = await asyncio.wait_for(transport.receive(), timeout=1)
            assert received == {"echo": {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}}}
        finally:
            await transport.close()


@pytest.mark.asyncio
async def test_client_transport_receive_returns_none_on_close() -> None:
    async def close_immediately(ws: Any) -> None:
        await ws.close()

    async with serve(close_immediately, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = await create_websocket_stream(f"ws://localhost:{port}")
        try:
            assert await asyncio.wait_for(transport.receive(), timeout=1) is None
        finally:
            await transport.close()


@pytest.mark.asyncio
async def test_client_transport_ignores_binary_frames() -> None:
    async def send_binary_then_text(ws: Any) -> None:
        await ws.send(b"\x00\x01")
        await ws.send(json.dumps({"ok": True}))
        await ws.recv()

    async with serve(send_binary_then_text, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = await create_websocket_stream(f"ws://localhost:{port}")
        try:
            received = await asyncio.wait_for(transport.receive(), timeout=1)
            assert received == {"ok": True}
        finally:
            await transport.close()


# -- ASGI websocket server handler ---------------------------------------------


class _FakeAsgiSocket:
    """In-memory ASGI websocket double driving handle_asgi_websocket."""

    def __init__(self) -> None:
        self._incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.accepted_headers: list[tuple[bytes, bytes]] = []
        self._sent_event = asyncio.Event()

    def client_connect(self) -> None:
        self._incoming.put_nowait({"type": "websocket.connect"})

    def client_send_text(self, message: dict[str, Any]) -> None:
        self._incoming.put_nowait({"type": "websocket.receive", "text": json.dumps(message)})

    def client_disconnect(self) -> None:
        self._incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})

    async def receive(self) -> dict[str, Any]:
        return await self._incoming.get()

    async def send(self, message: dict[str, Any]) -> None:
        if message["type"] == "websocket.accept":
            self.accepted_headers = message.get("headers", [])
        elif message["type"] == "websocket.send":
            self.sent.append(json.loads(message["text"]))
            self._sent_event.set()

    async def wait_for_send(self, predicate, timeout: float = 1.0) -> dict[str, Any]:
        async def _poll() -> dict[str, Any]:
            while True:
                for item in self.sent:
                    if predicate(item):
                        return item
                self._sent_event.clear()
                await self._sent_event.wait()

        return await asyncio.wait_for(_poll(), timeout=timeout)


@pytest.mark.asyncio
async def test_asgi_websocket_handshake_returns_connection_id() -> None:
    server = AcpServer(lambda conn: _Agent())
    socket = _FakeAsgiSocket()
    socket.client_connect()
    handler = asyncio.ensure_future(handle_asgi_websocket(server, {"type": "websocket"}, socket.receive, socket.send))
    await asyncio.sleep(0.05)
    header_names = [k for k, _ in socket.accepted_headers]
    assert CONNECTION_ID_HEADER.lower().encode() in header_names
    socket.client_disconnect()
    await asyncio.wait_for(handler, timeout=1)
    await server.close()


@pytest.mark.asyncio
async def test_asgi_websocket_full_flow() -> None:
    server = AcpServer(lambda conn: _Agent())
    socket = _FakeAsgiSocket()
    socket.client_connect()
    handler = asyncio.ensure_future(handle_asgi_websocket(server, {"type": "websocket"}, socket.receive, socket.send))
    await asyncio.sleep(0.05)

    socket.client_send_text({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 1}})
    init_resp = await socket.wait_for_send(lambda m: m.get("id") == 0)
    assert "result" in init_resp

    socket.client_send_text({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "session/new",
        "params": {"cwd": "/", "mcpServers": []},
    })
    new_resp = await socket.wait_for_send(lambda m: m.get("id") == 1)
    assert new_resp["result"]["sessionId"] == "sess-ws"

    socket.client_send_text({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "session/prompt",
        "params": {"sessionId": "sess-ws", "prompt": []},
    })
    notif = await socket.wait_for_send(lambda m: m.get("method") == "session/update")
    assert notif["params"]["sessionId"] == "sess-ws"
    prompt_resp = await socket.wait_for_send(lambda m: m.get("id") == 2)
    assert prompt_resp["result"]["stopReason"] == "end_turn"

    socket.client_disconnect()
    await asyncio.wait_for(handler, timeout=1)
    await server.close()
