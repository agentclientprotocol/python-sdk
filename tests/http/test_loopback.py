"""End-to-end in-process loopback tests: Python client transport <-> ASGI server.

Boots the ASGI app under a real uvicorn server (httpx's ASGITransport buffers
whole responses and cannot consume infinite SSE streams), then drives the full
ACP flow over both the Streamable HTTP and WebSocket transports.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from acp import connect_to_agent
from acp.http.asgi import create_asgi_app
from acp.http.client import create_http_stream
from acp.schema import InitializeResponse, NewSessionResponse, PromptResponse, RequestPermissionResponse
from acp.ws.client import create_websocket_stream
from tests.conftest import TestAgent, TestClient


class _LoopbackAgent(TestAgent):
    def __init__(self) -> None:
        super().__init__()
        self._conn: Any = None
        self.ask_permission = False

    def on_connect(self, conn: Any) -> None:
        self._conn = conn

    async def initialize(self, protocol_version: int = 1, **kwargs: Any) -> InitializeResponse:
        return InitializeResponse(protocol_version=1)

    async def new_session(self, cwd: str | None = None, mcp_servers: Any = None, **kwargs: Any) -> NewSessionResponse:
        return NewSessionResponse(session_id="sess-loop")

    async def prompt(self, session_id: str, prompt: Any = None, **kwargs: Any) -> PromptResponse:
        await self._conn.session_update(
            session_id=session_id,
            update={"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "hello"}},
        )
        if self.ask_permission:
            await self._conn.request_permission(
                session_id=session_id,
                tool_call={"toolCallId": "t1", "title": "run"},
                options=[{"optionId": "allow", "name": "Allow", "kind": "allow_once"}],
            )
        return PromptResponse(stop_reason="end_turn")


def _make_app(agent: _LoopbackAgent) -> Any:
    return create_asgi_app(lambda conn: agent)


class _CapturingClient(TestClient):
    def __init__(self) -> None:
        super().__init__()
        self.updates: list[Any] = []
        self.permission_requested = False

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append(update)

    async def request_permission(self, session_id: str, tool_call: Any, options: Any, **kwargs: Any):
        self.permission_requested = True
        return RequestPermissionResponse.model_validate({"outcome": {"outcome": "selected", "optionId": "allow"}})


# -- Streamable HTTP -----------------------------------------------------------


@pytest.mark.asyncio
async def test_http_loopback_initialize_and_new_session(serve_asgi) -> None:
    agent = _LoopbackAgent()
    server = await serve_asgi(_make_app(agent))
    transport = create_http_stream(server.http_url)
    conn = connect_to_agent(_CapturingClient(), transport)
    try:
        init = await asyncio.wait_for(conn.initialize(protocol_version=1), timeout=10)
        assert init.protocol_version == 1
        new = await asyncio.wait_for(conn.new_session(cwd=".", mcp_servers=[]), timeout=10)
        assert new.session_id == "sess-loop"
    finally:
        await conn.close()
        await transport.close()


@pytest.mark.asyncio
async def test_http_loopback_prompt_streams_and_permission(serve_asgi) -> None:
    agent = _LoopbackAgent()
    agent.ask_permission = True
    server = await serve_asgi(_make_app(agent))
    transport = create_http_stream(server.http_url)
    client = _CapturingClient()
    conn = connect_to_agent(client, transport)
    try:
        await asyncio.wait_for(conn.initialize(protocol_version=1), timeout=10)
        new = await asyncio.wait_for(conn.new_session(cwd=".", mcp_servers=[]), timeout=10)
        result = await asyncio.wait_for(conn.prompt(session_id=new.session_id, prompt=[]), timeout=10)
        assert result.stop_reason == "end_turn"
        await asyncio.sleep(0.2)
        assert client.updates, "expected a session/update notification over SSE"
        assert client.permission_requested, "expected a server->client permission request"
    finally:
        await conn.close()
        await transport.close()


# -- WebSocket -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_loopback_prompt_streams_and_permission(serve_asgi) -> None:
    agent = _LoopbackAgent()
    agent.ask_permission = True
    server = await serve_asgi(_make_app(agent))
    transport = await create_websocket_stream(server.ws_url)
    client = _CapturingClient()
    conn = connect_to_agent(client, transport)
    try:
        await asyncio.wait_for(conn.initialize(protocol_version=1), timeout=10)
        new = await asyncio.wait_for(conn.new_session(cwd=".", mcp_servers=[]), timeout=10)
        result = await asyncio.wait_for(conn.prompt(session_id=new.session_id, prompt=[]), timeout=10)
        assert result.stop_reason == "end_turn"
        await asyncio.sleep(0.2)
        assert client.updates, "expected a session/update notification over WS"
        assert client.permission_requested, "expected a server->client permission request"
    finally:
        await conn.close()
        await transport.close()
