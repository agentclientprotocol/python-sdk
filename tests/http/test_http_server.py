"""Tests for the framework-agnostic AcpServer core (ported from server*.test.ts)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from acp.http.protocol import CONNECTION_ID_HEADER
from acp.http.server import AcpServer
from acp.schema import NewSessionResponse, PromptResponse
from tests.conftest import TestAgent

CT_JSON = "application/json"


class _Agent(TestAgent):
    """A test agent that streams a notification during prompt and can ask permission."""

    def __init__(self) -> None:
        super().__init__()
        self._conn: Any = None
        self.ask_permission = False

    def on_connect(self, conn: Any) -> None:
        self._conn = conn

    async def new_session(self, cwd: str | None = None, mcp_servers: Any = None, **kwargs: Any) -> NewSessionResponse:
        return NewSessionResponse(session_id="sess-1")

    async def prompt(self, session_id: str, prompt: Any = None, **kwargs: Any) -> PromptResponse:
        # Emit a session-scoped notification back to the client.
        await self._conn.session_update(
            session_id=session_id,
            update={"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "hi"}},
        )
        if self.ask_permission:
            await self._conn.request_permission(
                session_id=session_id,
                tool_call={"toolCallId": "t1", "title": "run"},
                options=[{"optionId": "allow", "name": "Allow", "kind": "allow_once"}],
            )
        return PromptResponse(stop_reason="end_turn")


def _agent_factory(agent: _Agent):
    return lambda conn: agent


async def _drain_stream(server: AcpServer, connection_id: str, session_id: str | None, out: list[bytes]) -> None:
    async for frame in server.open_stream(connection_id=connection_id, session_id=session_id):
        out.append(frame)


@pytest.mark.asyncio
async def test_post_wrong_content_type_returns_415() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    result = await server.handle_post(
        {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}},
        content_type="text/plain",
        connection_id=None,
        session_id=None,
    )
    assert result.status == 415
    await server.close()


@pytest.mark.asyncio
async def test_batch_returns_501() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    result = await server.handle_post([], content_type=CT_JSON, connection_id=None, session_id=None)
    assert result.status == 501
    await server.close()


@pytest.mark.asyncio
async def test_initialize_creates_connection_and_returns_id() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    result = await server.handle_post(
        {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 1, "clientCapabilities": {}}},
        content_type=CT_JSON,
        connection_id=None,
        session_id=None,
    )
    assert result.status == 200
    assert CONNECTION_ID_HEADER in result.headers
    assert result.body is not None
    assert result.body["id"] == 0
    await server.close()


@pytest.mark.asyncio
async def test_missing_connection_id_returns_400() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    result = await server.handle_post(
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}},
        content_type=CT_JSON,
        connection_id=None,
        session_id=None,
    )
    assert result.status == 400
    await server.close()


@pytest.mark.asyncio
async def test_unknown_connection_id_returns_404() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    result = await server.handle_post(
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}},
        content_type=CT_JSON,
        connection_id="nope",
        session_id=None,
    )
    assert result.status == 404
    await server.close()


async def _initialize(server: AcpServer) -> str:
    result = await server.handle_post(
        {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 1, "clientCapabilities": {}}},
        content_type=CT_JSON,
        connection_id=None,
        session_id=None,
    )
    return result.headers[CONNECTION_ID_HEADER]


@pytest.mark.asyncio
async def test_session_new_result_on_connection_stream() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    conn_id = await _initialize(server)
    frames: list[bytes] = []
    task = asyncio.ensure_future(_drain_stream(server, conn_id, None, frames))
    await asyncio.sleep(0.05)
    result = await server.handle_post(
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": ".", "mcpServers": []}},
        content_type=CT_JSON,
        connection_id=conn_id,
        session_id=None,
    )
    assert result.status == 202
    await asyncio.sleep(0.1)
    joined = b"".join(frames).decode()
    assert '"sessionId":"sess-1"' in joined
    assert '"id":1' in joined
    task.cancel()
    await server.close()


@pytest.mark.asyncio
async def test_session_scoped_missing_session_header_returns_400() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    conn_id = await _initialize(server)
    result = await server.handle_post(
        {"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {"sessionId": "sess-1"}},
        content_type=CT_JSON,
        connection_id=conn_id,
        session_id=None,
    )
    assert result.status == 400
    await server.close()


@pytest.mark.asyncio
async def test_prompt_streams_notification_on_session_stream() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    conn_id = await _initialize(server)
    # Create the session first.
    await server.handle_post(
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": ".", "mcpServers": []}},
        content_type=CT_JSON,
        connection_id=conn_id,
        session_id=None,
    )
    await asyncio.sleep(0.05)
    session_frames: list[bytes] = []
    conn_frames: list[bytes] = []
    st = asyncio.ensure_future(_drain_stream(server, conn_id, "sess-1", session_frames))
    ct = asyncio.ensure_future(_drain_stream(server, conn_id, None, conn_frames))
    await asyncio.sleep(0.05)
    result = await server.handle_post(
        {"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {"sessionId": "sess-1", "prompt": []}},
        content_type=CT_JSON,
        connection_id=conn_id,
        session_id="sess-1",
    )
    assert result.status == 202
    await asyncio.sleep(0.15)
    session_joined = b"".join(session_frames).decode()
    # The agent_message_chunk notification is session-scoped.
    assert "agent_message_chunk" in session_joined
    # The prompt response (id 2) also routes to the session stream.
    assert '"id":2' in session_joined
    st.cancel()
    ct.cancel()
    await server.close()


@pytest.mark.asyncio
async def test_delete_terminates_connection() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    conn_id = await _initialize(server)
    result = await server.handle_delete(connection_id=conn_id)
    assert result.status == 202
    # Subsequent use of the connection id 404s.
    follow = await server.handle_post(
        {"jsonrpc": "2.0", "id": 5, "method": "session/new", "params": {}},
        content_type=CT_JSON,
        connection_id=conn_id,
        session_id=None,
    )
    assert follow.status == 404
    await server.close()


@pytest.mark.asyncio
async def test_delete_missing_connection_id_returns_400() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    result = await server.handle_delete(connection_id=None)
    assert result.status == 400
    await server.close()


@pytest.mark.asyncio
async def test_get_validation_errors() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    assert server.validate_stream(connection_id=None, session_id=None).status == 400  # type: ignore[union-attr]
    assert server.validate_stream(connection_id="nope", session_id=None).status == 404  # type: ignore[union-attr]
    conn_id = await _initialize(server)
    assert server.validate_stream(connection_id=conn_id, session_id="ghost").status == 404  # type: ignore[union-attr]
    assert server.validate_stream(connection_id=conn_id, session_id=None) is None
    await server.close()
