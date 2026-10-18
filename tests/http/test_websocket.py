"""WebSocket transport tests (client + ASGI server handler)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

import pytest
from starlette.websockets import WebSocket
from websockets.asyncio.server import serve
from websockets.protocol import State

from acp.connection import Connection
from acp.http.protocol import CONNECTION_ID_HEADER
from acp.schema import NewSessionResponse, PromptResponse
from acp.ws.client import create_websocket_stream
from acp.ws.server import handle_websocket
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


@pytest.mark.asyncio
@pytest.mark.parametrize("frame", ["null", "[]", "{}", "123", "1.5", '"x"', "true", "false", "invalid json", b"binary"])
async def test_client_transport_receives_valid_message_after_invalid_frame(frame: str | bytes) -> None:
    message = {"jsonrpc": "2.0", "id": 0, "result": {"items": [], "value": None}}

    async def send_frames(ws: Any) -> None:
        await ws.send(frame)
        await ws.send(json.dumps(message))
        await ws.close()

    async with serve(send_frames, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = await create_websocket_stream(f"ws://127.0.0.1:{port}")
        try:
            assert await asyncio.wait_for(transport.receive(), timeout=1) == message
            assert await asyncio.wait_for(transport.receive(), timeout=1) is None
        finally:
            await transport.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("frame", ["null", "[]", "{}"])
async def test_client_connection_recovers_after_invalid_frame(frame: str) -> None:
    async def respond(ws: Any) -> None:
        request = json.loads(await ws.recv())
        await ws.send(frame)
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {"protocolVersion": 1}}))
        await ws.wait_closed()

    async def handler(method: str, params: Any, is_notification: bool) -> None:
        pass

    async with serve(respond, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = await create_websocket_stream(f"ws://127.0.0.1:{port}")
        async with Connection(handler, transport) as conn:
            result = await asyncio.wait_for(conn.send_request("initialize", {"protocolVersion": 1}), timeout=1)
            assert result == {"protocolVersion": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame", "code", "message"),
    [
        ("invalid json", -32700, "Parse error"),
        ("null", -32600, "Invalid request"),
        ("[]", -32600, "Invalid request"),
        ("{}", -32600, "Invalid request"),
        ("123", -32600, "Invalid request"),
        ("1.5", -32600, "Invalid request"),
        ('"x"', -32600, "Invalid request"),
        ("true", -32600, "Invalid request"),
        ("false", -32600, "Invalid request"),
    ],
)
async def test_client_reports_protocol_error_then_receives_message(frame: str, code: int, message: str) -> None:
    valid = {"jsonrpc": "2.0", "id": "response", "result": None}
    replies = []

    async def respond(ws: Any) -> None:
        await ws.send(frame)
        replies.append(json.loads(await ws.recv()))
        await ws.send(json.dumps(valid))
        await ws.wait_closed()

    async with serve(respond, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = await create_websocket_stream(f"ws://127.0.0.1:{port}")
        try:
            assert await asyncio.wait_for(transport.receive(), timeout=1) == valid
            assert replies == [
                {"jsonrpc": "2.0", "id": None, "error": {"code": code, "message": message, "data": None}}
            ]
        finally:
            await transport.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [1000, 1011])
async def test_client_drains_buffered_response_when_error_reply_cannot_be_sent(code: int) -> None:
    closed = asyncio.Event()
    valid = {"jsonrpc": "2.0", "id": 0, "result": {"items": [], "value": None}}

    async def respond(ws: Any) -> None:
        await ws.send("invalid json")
        await ws.send("null")
        await ws.send(json.dumps(valid))
        await ws.close(code=code)
        closed.set()

    async with serve(respond, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = await create_websocket_stream(f"ws://127.0.0.1:{port}")
        try:
            await asyncio.wait_for(closed.wait(), timeout=1)
            assert await asyncio.wait_for(transport.receive(), timeout=1) == valid
            assert await asyncio.wait_for(transport.receive(), timeout=1) is None
        finally:
            await transport.close()


@pytest.mark.asyncio
async def test_client_drains_buffered_response_before_close_handshake_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    from websockets.asyncio.client import connect

    closing = asyncio.Event()
    release = asyncio.Event()
    valid = {"jsonrpc": "2.0", "id": 0, "result": None}
    connection: Any = None

    async def open_socket(url: str, **kwargs: Any) -> Any:
        nonlocal connection
        connection = await connect(url, **kwargs)
        return connection

    monkeypatch.setattr("acp.ws.client.ws_connect", open_socket)

    async def respond(ws: Any) -> None:
        await ws.send("null")
        await ws.send(json.dumps(valid))
        ws.transport.pause_reading()
        close = asyncio.create_task(ws.close())
        try:
            while ws.state == State.OPEN:
                await asyncio.sleep(0)
            closing.set()
            await release.wait()
        finally:
            ws.transport.resume_reading()
            await close

    async with serve(respond, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = await create_websocket_stream(f"ws://127.0.0.1:{port}")
        try:

            async def wait_for_closing() -> None:
                await closing.wait()
                while connection.state == State.OPEN:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_closing(), timeout=1)
            assert await asyncio.wait_for(transport.receive(), timeout=1) == valid
        finally:
            release.set()
            with contextlib.suppress(Exception):
                await transport.close()


# -- ASGI websocket server handler ---------------------------------------------


class _FakeAsgiSocket:
    """In-memory ASGI websocket double driving handle_websocket."""

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
    socket = _FakeAsgiSocket()
    socket.client_connect()
    handler = asyncio.ensure_future(
        handle_websocket(lambda conn: _Agent(), WebSocket({"type": "websocket"}, socket.receive, socket.send))
    )
    await asyncio.sleep(0.05)
    header_names = [k for k, _ in socket.accepted_headers]
    assert CONNECTION_ID_HEADER.lower().encode() in header_names
    socket.client_disconnect()
    await asyncio.wait_for(handler, timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect", [False, True])
async def test_asgi_websocket_cleans_up_running_prompt(disconnect: bool) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    class Agent(_Agent):
        async def prompt(self, session_id: str, prompt: Any = None, **kwargs: Any) -> PromptResponse:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            return PromptResponse(stop_reason="end_turn")

    socket = _FakeAsgiSocket()
    socket.client_connect()
    handler = asyncio.create_task(
        handle_websocket(lambda conn: Agent(), WebSocket({"type": "websocket"}, socket.receive, socket.send))
    )
    try:
        socket.client_send_text({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/prompt",
            "params": {"sessionId": "sess-ws", "prompt": []},
        })
        await asyncio.wait_for(started.wait(), timeout=1)
        if disconnect:
            socket.client_disconnect()
        else:
            handler.cancel()
        await asyncio.wait_for(handler, timeout=1)
        assert stopped.is_set()
    finally:
        socket.client_disconnect()
        await asyncio.wait_for(handler, timeout=1)


@pytest.mark.asyncio
async def test_asgi_websocket_reports_invalid_frames_and_ignores_binary() -> None:
    socket = _FakeAsgiSocket()
    socket.client_connect()
    frames = [b"binary", "invalid json", "[]", "{}", "null", "123", "1.5", '"x"', "true", "false"]
    for frame in frames:
        frame = {"bytes": frame} if isinstance(frame, bytes) else {"text": frame}
        socket._incoming.put_nowait({"type": "websocket.receive", **frame})
    handler = asyncio.create_task(
        handle_websocket(lambda conn: _Agent(), WebSocket({"type": "websocket"}, socket.receive, socket.send))
    )
    try:
        socket.client_send_text({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 1}})
        response = await socket.wait_for_send(lambda m: m.get("id") == 0)
        assert "result" in response
        assert [m["error"]["code"] for m in socket.sent if "error" in m] == [-32700, *([-32600] * 8)]
        assert all(m["id"] is None and m["jsonrpc"] == "2.0" for m in socket.sent if "error" in m)
    finally:
        socket.client_disconnect()
        await asyncio.wait_for(handler, timeout=1)


@pytest.mark.asyncio
async def test_asgi_websocket_delivers_notification_after_error_reply_fails() -> None:
    received = asyncio.Event()

    class Agent(_Agent):
        async def cancel(self, session_id: str, **kwargs: Any) -> None:
            assert session_id == "sess-ws"
            received.set()

    socket = _FakeAsgiSocket()
    socket.client_connect()

    async def failed_send(message: dict[str, Any]) -> None:
        if message["type"] == "websocket.send":
            raise OSError("Peer closed")
        await socket.send(message)

    for text in ("invalid json", "null", "{}"):
        socket._incoming.put_nowait({"type": "websocket.receive", "text": text})
    socket.client_send_text({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": "sess-ws"}})
    handler = asyncio.create_task(
        handle_websocket(lambda conn: Agent(), WebSocket({"type": "websocket"}, socket.receive, failed_send))
    )
    try:
        await asyncio.wait_for(received.wait(), timeout=1)
    finally:
        socket.client_disconnect()
        await asyncio.wait_for(handler, timeout=1)


@pytest.mark.asyncio
async def test_asgi_websocket_warns_once_on_binary_frames(caplog: pytest.LogCaptureFixture) -> None:
    socket = _FakeAsgiSocket()
    socket.client_connect()
    for _ in range(2):
        socket._incoming.put_nowait({"type": "websocket.receive", "bytes": b'{"jsonrpc":"2.0"}\n'})
    handler = asyncio.create_task(
        handle_websocket(lambda conn: _Agent(), WebSocket({"type": "websocket"}, socket.receive, socket.send))
    )
    try:
        with caplog.at_level(logging.WARNING):
            socket.client_send_text({
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {"protocolVersion": 1},
            })
            await socket.wait_for_send(lambda m: m.get("id") == 0)
        warnings = [r for r in caplog.records if "binary WebSocket frame" in r.getMessage()]
        assert len(warnings) == 1
    finally:
        socket.client_disconnect()
        await asyncio.wait_for(handler, timeout=1)


@pytest.mark.asyncio
async def test_asgi_websocket_full_flow() -> None:
    socket = _FakeAsgiSocket()
    socket.client_connect()
    handler = asyncio.ensure_future(
        handle_websocket(lambda conn: _Agent(), WebSocket({"type": "websocket"}, socket.receive, socket.send))
    )
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
