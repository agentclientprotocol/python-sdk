"""Tests for the AcpServer HTTP handlers (ported from server*.test.ts)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from acp.exceptions import RequestError
from acp.http.protocol import CONNECTION_ID_HEADER
from acp.http.server import AcpServer, _HttpTransport
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
    assert result.status_code == 415
    await server.close()


@pytest.mark.asyncio
async def test_batch_returns_501() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    result = await server.handle_post([], content_type=CT_JSON, connection_id=None, session_id=None)
    assert result.status_code == 501
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
    assert result.status_code == 200
    assert CONNECTION_ID_HEADER in result.headers
    assert result.body is not None
    assert json.loads(bytes(result.body))["id"] == 0
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
    assert result.status_code == 400
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
    assert result.status_code == 404
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
    assert result.status_code == 202
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
    assert result.status_code == 400
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
    assert result.status_code == 202
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
    assert result.status_code == 202
    # Subsequent use of the connection id 404s.
    follow = await server.handle_post(
        {"jsonrpc": "2.0", "id": 5, "method": "session/new", "params": {}},
        content_type=CT_JSON,
        connection_id=conn_id,
        session_id=None,
    )
    assert follow.status_code == 404
    await server.close()


@pytest.mark.asyncio
async def test_delete_missing_connection_id_returns_400() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    result = await server.handle_delete(connection_id=None)
    assert result.status_code == 400
    await server.close()


@pytest.mark.asyncio
async def test_get_validation_errors() -> None:
    server = AcpServer(_agent_factory(_Agent()))
    assert server.validate_stream(connection_id=None, session_id=None).status_code == 400  # type: ignore[union-attr]
    assert server.validate_stream(connection_id="nope", session_id=None).status_code == 404  # type: ignore[union-attr]
    conn_id = await _initialize(server)
    assert server.validate_stream(connection_id=conn_id, session_id="ghost").status_code == 404  # type: ignore[union-attr]
    assert server.validate_stream(connection_id=conn_id, session_id=None) is None
    await server.close()


@pytest.mark.asyncio
async def test_concurrent_session_results_and_errors_stay_on_their_streams() -> None:
    class Agent(_Agent):
        async def new_session(
            self, cwd: str | None = None, mcp_servers: Any = None, **kwargs: Any
        ) -> NewSessionResponse:
            return NewSessionResponse(session_id=str(cwd))

        async def prompt(self, session_id: str, prompt: Any = None, **kwargs: Any) -> PromptResponse:
            if session_id == "first":
                raise RequestError(-32000, "prompt failed")
            return PromptResponse(stop_reason="end_turn")

    server = AcpServer(lambda conn: Agent())
    connection_id = await _initialize(server)
    connection_stream = server.open_stream(connection_id=connection_id, session_id=None)
    try:
        for request_id, session_id in enumerate(("first", "second")):
            # Reuse initialize's id=0: its completed waiter must not intercept this.
            await server.handle_post(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "session/new",
                    "params": {"cwd": session_id, "mcpServers": []},
                },
                content_type=CT_JSON,
                connection_id=connection_id,
                session_id=None,
            )
            frame = await asyncio.wait_for(anext(connection_stream), timeout=1)
            assert json.loads(frame.removeprefix(b"data: "))["result"]["sessionId"] == session_id

        for request_id, session_id in enumerate(("first", "second"), start=2):
            result = await server.handle_post(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "session/prompt",
                    "params": {"sessionId": session_id, "prompt": []},
                },
                content_type=CT_JSON,
                connection_id=connection_id,
                session_id=session_id,
            )
            assert result.status_code == 202

        for request_id, session_id in enumerate(("first", "second"), start=2):
            stream = server.open_stream(connection_id=connection_id, session_id=session_id)
            try:
                frame = await asyncio.wait_for(anext(stream), timeout=1)
                response = json.loads(frame.removeprefix(b"data: "))
                assert response["id"] == request_id
                if session_id == "first":
                    assert response["error"]["message"] == "prompt failed"
                else:
                    assert response["result"]["stopReason"] == "end_turn"
            finally:
                await stream.aclose()
    finally:
        await connection_stream.aclose()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_succeeds", [False, True])
@pytest.mark.parametrize("second_succeeds", [False, True])
async def test_overlapping_loads_preserve_successful_session_streams(
    first_succeeds: bool, second_succeeds: bool
) -> None:
    transport = _HttpTransport(0)
    connection_stream = transport.connection_stream.iterate()
    try:
        for request_id in (1, 2):
            await transport.deliver_to_agent({
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "session/load",
                "params": {"sessionId": "saved", "cwd": "/", "mcpServers": []},
            })
        # A client may attach its session GET before load completes.
        session_stream = transport.session_streams["saved"]
        for request_id, succeeds in enumerate((first_succeeds, second_succeeds), start=1):
            response = {"jsonrpc": "2.0", "id": request_id}
            response.update({"result": {}} if succeeds else {"error": {"code": -32000, "message": "load failed"}})
            await transport.send(response)
            assert await asyncio.wait_for(anext(connection_stream), timeout=1) == response
            if request_id == 1:
                assert transport.session_streams["saved"] is session_stream

        if first_succeeds or second_succeeds:
            assert transport.session_streams["saved"] is session_stream
            live = {"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "saved"}}
            await transport.send(live)
            messages = session_stream.iterate()
            assert await asyncio.wait_for(anext(messages), timeout=1) == live
            await messages.aclose()
        else:
            assert "saved" not in transport.session_streams
            assert [message async for message in session_stream.iterate()] == []
    finally:
        await connection_stream.aclose()
        await transport.close()
