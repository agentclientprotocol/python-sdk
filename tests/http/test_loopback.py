"""End-to-end in-process loopback tests: Python client transport <-> ASGI server.

Boots the ASGI app under a real uvicorn server (httpx's ASGITransport buffers
whole responses and cannot consume infinite SSE streams), then drives the full
ACP flow over both the Streamable HTTP and WebSocket transports.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from acp import RequestError, connect_to_agent
from acp.http.asgi import create_asgi_app
from acp.http.client import create_http_stream
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    RequestPermissionResponse,
    TextContentBlock,
)
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


class _LoadingAgent(_LoopbackAgent):
    def __init__(self, history_size: int) -> None:
        super().__init__()
        self.history_size = history_size
        self.fail_load = False
        self.ask_permission = True

    async def initialize(self, protocol_version: int = 1, **kwargs: Any) -> InitializeResponse:
        return InitializeResponse(protocol_version=1, agent_capabilities=AgentCapabilities(load_session=True))

    async def load_session(self, cwd: str, session_id: str, **kwargs: Any) -> LoadSessionResponse:
        for i in range(self.history_size):
            await self._conn.session_update(
                session_id=session_id,
                update=AgentMessageChunk(content=TextContentBlock(text=f"history-{i}")),
            )
        if self.fail_load:
            raise RequestError(-32000, "load failed")
        return LoadSessionResponse()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["http", "ws"])
@pytest.mark.parametrize("history_size", [0, 1100])
async def test_load_session_replays_history_and_supports_prompt(protocol: str, history_size: int, serve_asgi) -> None:
    agent = _LoadingAgent(history_size)
    server = await serve_asgi(_make_app(agent))
    transport = (
        create_http_stream(server.http_url) if protocol == "http" else await create_websocket_stream(server.ws_url)
    )
    client = _CapturingClient()
    conn = connect_to_agent(client, transport)
    try:
        init = await conn.initialize(protocol_version=1)
        assert init.agent_capabilities.load_session
        # Load an existing ID without first calling session/new, then reload it.
        for _ in range(2):
            client.updates.clear()
            loaded = await asyncio.wait_for(conn.load_session(cwd="/", session_id="saved-session"), timeout=10)
            assert loaded == LoadSessionResponse()
            assert [update.content.text for update in client.updates] == [f"history-{i}" for i in range(history_size)]
            result = await asyncio.wait_for(conn.prompt(session_id="saved-session", prompt=[]), timeout=10)
            assert result.stop_reason == "end_turn"
            assert client.permission_requested
            assert client.updates[-1].content.text == "hello"
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["http", "ws"])
async def test_failed_load_can_retry_and_preserves_existing_session(protocol: str, serve_asgi) -> None:
    agent = _LoadingAgent(1)
    server = await serve_asgi(_make_app(agent))
    transport = (
        create_http_stream(server.http_url) if protocol == "http" else await create_websocket_stream(server.ws_url)
    )
    conn = connect_to_agent(_CapturingClient(), transport)
    try:
        await conn.initialize(protocol_version=1)
        agent.fail_load = True
        with pytest.raises(RequestError, match="load failed"):
            await asyncio.wait_for(conn.load_session(cwd="/", session_id="saved-session"), timeout=5)
        agent.fail_load = False
        agent.history_size = 0
        await asyncio.wait_for(conn.load_session(cwd="/", session_id="saved-session"), timeout=5)
        agent.fail_load = True
        with pytest.raises(RequestError, match="load failed"):
            await asyncio.wait_for(conn.load_session(cwd="/", session_id="saved-session"), timeout=5)
        result = await asyncio.wait_for(conn.prompt(session_id="saved-session", prompt=[]), timeout=5)
        assert result.stop_reason == "end_turn"
    finally:
        await conn.close()
