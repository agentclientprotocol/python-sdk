from __future__ import annotations

import asyncio
from typing import Any

import pytest

import acp
from acp._transport import memory_transport_pair
from acp.connection import Connection, StreamDirection, StreamEvent
from acp.experimental import AgentProtocolRouter, v2
from acp.experimental.v2.meta import AGENT_METHODS


class Client:
    pass


class V1Client(acp.Client):
    pass


class V1Agent:
    def __init__(self) -> None:
        self.initialize_calls = 0
        self.client_name: str | None = None

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: acp.schema.ClientCapabilities | None = None,
        client_info: acp.schema.Implementation | None = None,
        **kwargs: Any,
    ) -> acp.InitializeResponse:
        self.initialize_calls += 1
        self.client_name = client_info.name if client_info is not None else None
        return acp.InitializeResponse(protocol_version=protocol_version)


class V2Agent:
    def __init__(self) -> None:
        self.initialize_calls = 0

    async def initialize(self, request: v2.schema.InitializeRequest) -> v2.schema.InitializeResponse:
        self.initialize_calls += 1
        return v2.schema.InitializeResponse(
            protocol_version=v2.PROTOCOL_VERSION,
            info=v2.schema.Implementation(name="v2-agent", version="1.0.0"),
        )


class UpdateClient:
    def __init__(self) -> None:
        self.updates: asyncio.Queue[v2.schema.UpdateSessionNotification] = asyncio.Queue()

    async def session_update(self, notification: v2.schema.UpdateSessionNotification) -> None:
        await self.updates.put(notification)


class RoutedV2Agent(V2Agent):
    def __init__(self, connection: v2.AgentSideConnection) -> None:
        super().__init__()
        self.connection = connection

    async def prompt(self, request: v2.schema.PromptRequest) -> v2.schema.PromptResponse:
        await self.connection.session_update(
            v2.schema.UpdateSessionNotification(
                session_id=request.session_id,
                update=v2.schema.IdleSessionStateUpdate(),
            )
        )
        return v2.schema.PromptResponse()


def v2_initialize() -> v2.schema.InitializeRequest:
    return v2.schema.InitializeRequest(
        protocol_version=v2.PROTOCOL_VERSION,
        info=v2.schema.Implementation(name="v2-client", version="2.0.0"),
    )


@pytest.mark.asyncio
async def test_agent_protocol_router_selects_v2() -> None:
    client_transport, agent_transport = memory_transport_pair()
    v1_agent = V1Agent()
    v2_agent = V2Agent()
    wire: list[StreamEvent] = []
    agent_connection = AgentProtocolRouter(v1=lambda _: v1_agent, v2=lambda _: v2_agent).connect(agent_transport)
    client_connection = v2.ClientSideConnection(Client(), client_transport, observers=[wire.append])

    try:
        initialized = await client_connection.initialize(v2_initialize())

        assert initialized.info.name == "v2-agent"
        assert v1_agent.initialize_calls == 0
        assert v2_agent.initialize_calls == 1
        assert _initialize_count(wire) == 1
    finally:
        await client_connection.close()
        await agent_connection.close()


@pytest.mark.asyncio
async def test_agent_protocol_router_selects_v1() -> None:
    client_transport, agent_transport = memory_transport_pair()
    v1_agent = V1Agent()
    v2_agent = V2Agent()
    wire: list[StreamEvent] = []
    agent_connection = AgentProtocolRouter(v1=lambda _: v1_agent, v2=lambda _: v2_agent).connect(agent_transport)
    client_connection = acp.connect_to_agent(V1Client(), client_transport, observers=[wire.append])

    try:
        initialized = await client_connection.initialize(
            protocol_version=acp.PROTOCOL_VERSION,
            client_info=acp.schema.Implementation(name="v1-client", version="1.0.0"),
        )

        assert initialized.protocol_version == acp.PROTOCOL_VERSION
        assert v1_agent.initialize_calls == 1
        assert v1_agent.client_name == "v1-client"
        assert v2_agent.initialize_calls == 0
        assert _initialize_count(wire) == 1
    finally:
        await client_connection.close()
        await agent_connection.close()


@pytest.mark.asyncio
async def test_agent_protocol_router_normalizes_v2_initialize_for_v1() -> None:
    client_transport, agent_transport = memory_transport_pair()
    agent = V1Agent()
    wire: list[StreamEvent] = []
    agent_connection = AgentProtocolRouter(v1=lambda _: agent).connect(agent_transport)

    async def ignore_incoming(method: str, params: Any, is_notification: bool) -> None:
        pass

    client_connection = Connection(ignore_incoming, client_transport, observers=[wire.append])

    try:
        response = await client_connection.send_request(
            AGENT_METHODS["initialize"],
            v2_initialize().model_dump(mode="json", by_alias=True, exclude_none=True),
        )

        assert response["protocolVersion"] == acp.PROTOCOL_VERSION
        assert agent.initialize_calls == 1
        assert agent.client_name == "v2-client"
        assert _initialize_count(wire) == 1
    finally:
        await client_connection.close()
        await agent_connection.close()


@pytest.mark.asyncio
async def test_agent_protocol_router_isolates_connections() -> None:
    agents: list[RoutedV2Agent] = []

    def create_agent(connection: v2.AgentSideConnection) -> RoutedV2Agent:
        agent = RoutedV2Agent(connection)
        agents.append(agent)
        return agent

    router = AgentProtocolRouter(v2=create_agent)
    client_1_transport, agent_1_transport = memory_transport_pair()
    client_2_transport, agent_2_transport = memory_transport_pair()
    client_1 = UpdateClient()
    client_2 = UpdateClient()
    agent_connection_1 = router.connect(agent_1_transport)
    agent_connection_2 = router.connect(agent_2_transport)
    client_connection_1 = v2.ClientSideConnection(client_1, client_1_transport)
    client_connection_2 = v2.ClientSideConnection(client_2, client_2_transport)

    try:
        await client_connection_1.initialize(v2_initialize())
        await client_connection_2.initialize(v2_initialize())
        await client_connection_1.prompt(
            v2.schema.PromptRequest(
                session_id="session-1",
                prompt=[v2.schema.TextContentBlock(text="hello")],
            )
        )

        update = await asyncio.wait_for(client_1.updates.get(), timeout=1)
        assert update.session_id == "session-1"
        assert client_2.updates.empty()
        assert len(agents) == 2
        assert agents[0] is not agents[1]
    finally:
        await client_connection_1.close()
        await client_connection_2.close()
        await agent_connection_1.close()
        await agent_connection_2.close()


def _initialize_count(events: list[StreamEvent]) -> int:
    return sum(
        event.direction == StreamDirection.OUTGOING and event.message.get("method") == "initialize" for event in events
    )
