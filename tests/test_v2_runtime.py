from __future__ import annotations

from typing import Any

import pytest

from acp._transport import memory_transport_pair
from acp.exceptions import RequestError
from acp.experimental import v2


class Client:
    pass


class Agent:
    def __init__(self, *, response_version: int = v2.PROTOCOL_VERSION) -> None:
        self.response_version = response_version
        self.initialize_calls = 0

    async def initialize(self, request: v2.schema.InitializeRequest) -> v2.schema.InitializeResponse:
        self.initialize_calls += 1
        return v2.schema.InitializeResponse(
            protocol_version=self.response_version,
            info=v2.schema.Implementation(name="test-agent", version="1.0.0"),
            capabilities=v2.schema.AgentCapabilities(session=v2.schema.SessionCapabilities()),
        )

    async def new_session(self, request: v2.schema.NewSessionRequest) -> v2.schema.NewSessionResponse:
        return v2.schema.NewSessionResponse(session_id=f"session:{request.cwd}")


class SessionAgent(Agent):
    def on_connect(self, connection: v2.AgentSideConnection) -> None:
        self.connection = connection

    async def new_session(self, request: v2.schema.NewSessionRequest) -> v2.schema.NewSessionResponse:
        response = await super().new_session(request)
        await self.connection.session_update(
            v2.schema.UpdateSessionNotification(
                session_id=response.session_id,
                update=v2.schema.IdleSessionStateUpdate(),
            )
        )
        return response

    async def prompt(self, request: v2.schema.PromptRequest) -> v2.schema.PromptResponse:
        await self.connection.session_update(
            v2.schema.UpdateSessionNotification(
                session_id=request.session_id,
                update=v2.schema.RunningSessionStateUpdate(),
            )
        )
        await self.connection.session_update(
            v2.schema.UpdateSessionNotification(
                session_id=request.session_id,
                update=v2.schema.IdleSessionStateUpdate(stop_reason="end_turn"),
            )
        )
        return v2.schema.PromptResponse()


def initialize_request(protocol_version: int = v2.PROTOCOL_VERSION) -> v2.schema.InitializeRequest:
    return v2.schema.InitializeRequest(
        protocol_version=protocol_version,
        info=v2.schema.Implementation(name="test-client", version="1.0.0"),
    )


@pytest.mark.asyncio
async def test_v2_runtime_initializes_and_routes_generated_models() -> None:
    client_transport, agent_transport = memory_transport_pair()
    agent = Agent()
    agent_connection = v2.AgentSideConnection(agent, agent_transport)
    client_connection = v2.ClientSideConnection(Client(), client_transport)

    try:
        initialized = await client_connection.initialize(initialize_request())
        session = await client_connection.new_session(v2.schema.NewSessionRequest(cwd="/workspace"))

        assert initialized.protocol_version == v2.PROTOCOL_VERSION
        assert session.session_id == "session:/workspace"
        assert agent.initialize_calls == 1
        assert (await agent_connection.wait_until_initialized()).request.info.name == "test-client"
    finally:
        await client_connection.close()
        await agent_connection.close()


@pytest.mark.asyncio
async def test_v2_runtime_rejects_calls_before_initialize() -> None:
    client_transport, agent_transport = memory_transport_pair()
    agent_connection = v2.AgentSideConnection(Agent(), agent_transport)
    client_connection = v2.ClientSideConnection(Client(), client_transport)

    try:
        with pytest.raises(RequestError, match="Invalid request"):
            await client_connection.new_session(v2.schema.NewSessionRequest(cwd="/workspace"))
    finally:
        await client_connection.close()
        await agent_connection.close()


@pytest.mark.asyncio
async def test_v2_runtime_rejects_a_different_protocol_version() -> None:
    client_transport, agent_transport = memory_transport_pair()
    agent = Agent()
    agent_connection = v2.AgentSideConnection(agent, agent_transport)
    client_connection = v2.ClientSideConnection(Client(), client_transport)

    try:
        with pytest.raises(RequestError) as error:
            await client_connection.initialize(initialize_request(protocol_version=1))

        assert isinstance(error.value, RequestError)
        assert error.value.code == -32602
        assert agent.initialize_calls == 0
    finally:
        await client_connection.close()
        await agent_connection.close()


@pytest.mark.asyncio
async def test_v2_runtime_rejects_a_mismatched_initialize_response() -> None:
    client_transport, agent_transport = memory_transport_pair()
    agent_connection = v2.AgentSideConnection(Agent(response_version=1), agent_transport)
    client_connection = v2.ClientSideConnection(Client(), client_transport)

    try:
        with pytest.raises(RequestError) as error:
            await client_connection.initialize(initialize_request())

        assert isinstance(error.value, RequestError)
        assert error.value.code == -32600
    finally:
        await client_connection.close()
        await agent_connection.close()


@pytest.mark.asyncio
async def test_active_session_completes_only_after_running_then_idle() -> None:
    client_transport, agent_transport = memory_transport_pair()
    agent_connection = v2.AgentSideConnection(SessionAgent(), agent_transport)
    client_connection = v2.ClientSideConnection(Client(), client_transport)

    try:
        await client_connection.initialize(initialize_request())
        session = await client_connection.open_session(v2.schema.NewSessionRequest(cwd="/workspace"))
        await session.prompt(
            v2.schema.PromptRequest(
                session_id=session.session_id,
                prompt=[v2.schema.TextContentBlock(text="hello")],
            )
        )

        ready = await session.next_update()
        running = await session.next_update()
        stopped = await session.next_update()

        assert isinstance(ready, v2.SessionUpdate)
        assert isinstance(ready.update, v2.schema.IdleSessionStateUpdate)
        assert isinstance(running, v2.SessionUpdate)
        assert isinstance(running.update, v2.schema.RunningSessionStateUpdate)
        assert isinstance(stopped, v2.SessionStop)
        assert stopped.stop_reason == "end_turn"
        session.dispose()
    finally:
        await client_connection.close()
        await agent_connection.close()


def test_v2_public_entry_point_is_explicit() -> None:
    exported: dict[str, Any] = {name: getattr(v2, name) for name in v2.__all__}

    assert exported["PROTOCOL_VERSION"] == 2
    assert exported["schema"] is v2.schema
    assert "InitializeRequest" not in v2.__all__
