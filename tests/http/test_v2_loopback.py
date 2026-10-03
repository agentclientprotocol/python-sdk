"""v2 end-to-end loopback over the version-agnostic HTTP/WS transports.

The v2 runtime is served through an :class:`AgentProtocolRouter` instead of a
v1 ``AgentFactory``; the ASGI server must negotiate v2 from ``initialize`` and
route ``session/update`` traffic (and ``session/resume`` replay) the same way it
already routes v1's ``session/load``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import acp
from acp.experimental import AgentProtocolRouter, v2
from acp.http.asgi import create_asgi_app
from acp.http.client import create_http_stream
from acp.ws.client import create_websocket_stream
from tests.conftest import TestAgent, TestClient


class RoutedV2Agent:
    def __init__(self) -> None:
        self.connection: v2.AgentSideConnection | None = None

    def on_connect(self, connection: v2.AgentSideConnection) -> None:
        self.connection = connection

    async def initialize(
        self,
        protocol_version: int,
        info: v2.schema.Implementation,
        capabilities: v2.schema.ClientCapabilities | None = None,
        **kwargs: Any,
    ) -> v2.schema.InitializeResponse:
        return v2.schema.InitializeResponse(
            protocol_version=v2.PROTOCOL_VERSION,
            info=v2.schema.Implementation(name="v2-agent", version="1.0.0"),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> v2.schema.NewSessionResponse:
        return v2.schema.NewSessionResponse(session_id="sess-v2")

    async def prompt(
        self,
        session_id: str,
        prompt: list[Any],
        **kwargs: Any,
    ) -> v2.schema.PromptResponse:
        assert self.connection is not None
        await self.connection.session_update(
            session_id=session_id,
            update=v2.schema.AgentMessageChunk(
                message_id="msg-1",
                content=v2.schema.TextContentBlock(text="hello-v2"),
            ),
        )
        return v2.schema.PromptResponse(message_id="user-msg-1")

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        replay_from: Any = None,
        **kwargs: Any,
    ) -> v2.schema.ResumeSessionResponse:
        assert self.connection is not None
        for i in range(3):
            await self.connection.session_update(
                session_id=session_id,
                update=v2.schema.AgentMessageChunk(
                    message_id=f"history-{i}",
                    content=v2.schema.TextContentBlock(text=f"history-{i}"),
                ),
            )
        return v2.schema.ResumeSessionResponse()


class CapturingClient:
    def __init__(self) -> None:
        self.updates: list[tuple[str, Any]] = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append((session_id, update))


def _make_app(router: AgentProtocolRouter) -> Any:
    return create_asgi_app(router)


def _v2_agent_factory(connection: v2.AgentSideConnection) -> RoutedV2Agent:
    return RoutedV2Agent()


def _router() -> AgentProtocolRouter:
    return AgentProtocolRouter(v2=_v2_agent_factory)


async def _v2_client(transport: Any) -> tuple[v2.ClientSideConnection, CapturingClient]:
    client = CapturingClient()
    conn = v2.connect_to_agent(client, transport)
    return conn, client


@pytest.mark.asyncio
async def test_v2_http_prompt_streams_update(serve_asgi) -> None:
    server = await serve_asgi(_make_app(_router()))
    transport = create_http_stream(server.http_url)
    conn, client = await _v2_client(transport)
    try:
        init = await asyncio.wait_for(
            conn.initialize(
                protocol_version=v2.PROTOCOL_VERSION,
                info=v2.schema.Implementation(name="v2-client", version="1.0.0"),
            ),
            timeout=10,
        )
        assert init.protocol_version == v2.PROTOCOL_VERSION
        new = await asyncio.wait_for(conn.new_session(cwd="."), timeout=10)
        assert new.session_id == "sess-v2"
        result = await asyncio.wait_for(
            conn.prompt(session_id=new.session_id, prompt=[v2.schema.TextContentBlock(text="hi")]),
            timeout=10,
        )
        assert result.message_id == "user-msg-1"
        await asyncio.sleep(0.2)
        assert client.updates, "expected a session/update over SSE"
        assert client.updates[0][0] == "sess-v2"
        assert client.updates[0][1].content.text == "hello-v2"
    finally:
        await conn.close()
        await transport.close()


@pytest.mark.asyncio
async def test_v2_websocket_prompt_streams_update(serve_asgi) -> None:
    server = await serve_asgi(_make_app(_router()))
    transport = await create_websocket_stream(server.ws_url)
    conn, client = await _v2_client(transport)
    try:
        await asyncio.wait_for(
            conn.initialize(
                protocol_version=v2.PROTOCOL_VERSION,
                info=v2.schema.Implementation(name="v2-client", version="1.0.0"),
            ),
            timeout=10,
        )
        new = await asyncio.wait_for(conn.new_session(cwd="."), timeout=10)
        await asyncio.wait_for(
            conn.prompt(session_id=new.session_id, prompt=[v2.schema.TextContentBlock(text="hi")]),
            timeout=10,
        )
        await asyncio.sleep(0.2)
        assert client.updates, "expected a session/update over WS"
        assert client.updates[0][1].content.text == "hello-v2"
    finally:
        await conn.close()
        await transport.close()


@pytest.mark.asyncio
async def test_v2_http_resume_replays_history(serve_asgi) -> None:
    server = await serve_asgi(_make_app(_router()))
    transport = create_http_stream(server.http_url)
    conn, client = await _v2_client(transport)
    try:
        await asyncio.wait_for(
            conn.initialize(
                protocol_version=v2.PROTOCOL_VERSION,
                info=v2.schema.Implementation(name="v2-client", version="1.0.0"),
            ),
            timeout=10,
        )
        loaded = await asyncio.wait_for(
            conn.resume_session(session_id="saved-session", cwd="/"),
            timeout=10,
        )
        assert loaded == v2.schema.ResumeSessionResponse()
        assert [update.content.text for _, update in client.updates] == [f"history-{i}" for i in range(3)]
    finally:
        await conn.close()
        await transport.close()


@pytest.mark.asyncio
async def test_dual_stack_http_serves_v1_and_v2(serve_asgi) -> None:
    router = AgentProtocolRouter(v1=lambda conn: TestAgent(), v2=_v2_agent_factory)
    server = await serve_asgi(create_asgi_app(router))

    v1_transport = create_http_stream(server.http_url)
    v1_conn = acp.connect_to_agent(TestClient(), v1_transport)
    try:
        init = await asyncio.wait_for(v1_conn.initialize(protocol_version=acp.PROTOCOL_VERSION), timeout=10)
        assert init.protocol_version == acp.PROTOCOL_VERSION
        new = await asyncio.wait_for(v1_conn.new_session(cwd="."), timeout=10)
        assert new.session_id == "test-session-123"
    finally:
        await v1_conn.close()
        await v1_transport.close()

    v2_transport = create_http_stream(server.http_url)
    v2_conn, _ = await _v2_client(v2_transport)
    try:
        init = await asyncio.wait_for(
            v2_conn.initialize(
                protocol_version=v2.PROTOCOL_VERSION,
                info=v2.schema.Implementation(name="v2-client", version="1.0.0"),
            ),
            timeout=10,
        )
        assert init.protocol_version == v2.PROTOCOL_VERSION
        new = await asyncio.wait_for(v2_conn.new_session(cwd="."), timeout=10)
        assert new.session_id == "sess-v2"
    finally:
        await v2_conn.close()
        await v2_transport.close()
