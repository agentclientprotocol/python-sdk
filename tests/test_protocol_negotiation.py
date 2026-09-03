from __future__ import annotations

from typing import Any, cast

import pytest

import acp
from acp._transport import memory_transport_pair
from acp.connection import Connection, StreamDirection, StreamEvent
from acp.experimental import AgentProtocolRouter, v2


class Client:
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
    agent_connection = AgentProtocolRouter(v1=v1_agent, v2=v2_agent).connect(agent_transport)
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
    agent_connection = AgentProtocolRouter(v1=v1_agent, v2=v2_agent).connect(agent_transport)
    client_connection = acp.connect_to_agent(cast(acp.Client, Client()), client_transport, observers=[wire.append])

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
    agent_connection = AgentProtocolRouter(v1=agent).connect(agent_transport)

    async def ignore_incoming(method: str, params: Any, is_notification: bool) -> None:
        pass

    client_connection = Connection(ignore_incoming, client_transport, observers=[wire.append])

    try:
        response = await client_connection.send_request(
            v2.AGENT_METHODS["initialize"],
            v2_initialize().model_dump(mode="json", by_alias=True, exclude_none=True),
        )

        assert response["protocolVersion"] == acp.PROTOCOL_VERSION
        assert agent.initialize_calls == 1
        assert agent.client_name == "v2-client"
        assert _initialize_count(wire) == 1
    finally:
        await client_connection.close()
        await agent_connection.close()


def _initialize_count(events: list[StreamEvent]) -> int:
    return sum(
        event.direction == StreamDirection.OUTGOING and event.message.get("method") == "initialize" for event in events
    )
