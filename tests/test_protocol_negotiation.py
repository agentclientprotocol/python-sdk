from __future__ import annotations

from typing import Any

import pytest

import acp
from acp._transport import memory_transport_pair
from acp.connection import StreamDirection, StreamEvent
from acp.experimental import (
    AgentProtocolRouter,
    ClientNegotiator,
    NegotiatedV1,
    NegotiatedV2,
    V1ClientConfig,
    V2ClientConfig,
    v2,
)


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

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> acp.NewSessionResponse:
        return acp.NewSessionResponse(session_id=f"v1:{cwd}")


class V2Agent:
    def __init__(self) -> None:
        self.initialize_calls = 0

    async def initialize(self, request: v2.schema.InitializeRequest) -> v2.schema.InitializeResponse:
        self.initialize_calls += 1
        return v2.schema.InitializeResponse(
            protocol_version=v2.PROTOCOL_VERSION,
            info=v2.schema.Implementation(name="v2-agent", version="1.0.0"),
        )


def v1_config() -> V1ClientConfig:
    return V1ClientConfig(
        client=Client(),
        initialize=acp.InitializeRequest(
            protocol_version=acp.PROTOCOL_VERSION,
            client_info=acp.schema.Implementation(name="v1-client", version="1.0.0"),
        ),
    )


def v2_config() -> V2ClientConfig:
    return V2ClientConfig(
        client=Client(),
        initialize=v2.schema.InitializeRequest(
            protocol_version=v2.PROTOCOL_VERSION,
            info=v2.schema.Implementation(name="v2-client", version="2.0.0"),
        ),
    )


@pytest.mark.asyncio
async def test_negotiation_selects_v2_with_one_initialize() -> None:
    client_transport, agent_transport = memory_transport_pair()
    v1_agent = V1Agent()
    v2_agent = V2Agent()
    wire: list[StreamEvent] = []
    router = AgentProtocolRouter(v1=v1_agent, v2=v2_agent)
    agent_connection = router.connect(agent_transport)
    negotiator = ClientNegotiator(
        client_transport,
        v1=v1_config(),
        v2=v2_config(),
        observers=[wire.append],
    )

    try:
        negotiated = await negotiator.negotiate()
        repeated = await negotiator.negotiate()

        assert isinstance(negotiated, NegotiatedV2)
        assert repeated is negotiated
        assert negotiated.initialize.info.name == "v2-agent"
        assert v1_agent.initialize_calls == 0
        assert v2_agent.initialize_calls == 1
        assert _initialize_count(wire) == 1
    finally:
        await negotiator.close()
        await agent_connection.close()


@pytest.mark.asyncio
async def test_negotiation_downgrades_v2_initialize_without_repeating_it() -> None:
    client_transport, agent_transport = memory_transport_pair()
    agent = V1Agent()
    wire: list[StreamEvent] = []
    router = AgentProtocolRouter(v1=agent)
    agent_connection = router.connect(agent_transport)
    negotiator = ClientNegotiator(
        client_transport,
        v1=v1_config(),
        v2=v2_config(),
        observers=[wire.append],
    )

    try:
        negotiated = await negotiator.negotiate()

        assert isinstance(negotiated, NegotiatedV1)
        assert negotiated.initialize.protocol_version == acp.PROTOCOL_VERSION
        assert agent.initialize_calls == 1
        assert agent.client_name == "v2-client"
        assert _initialize_count(wire) == 1

        session = await negotiated.connection.new_session(cwd="/workspace")
        assert session.session_id == "v1:/workspace"
    finally:
        await negotiator.close()
        await agent_connection.close()


def _initialize_count(events: list[StreamEvent]) -> int:
    return sum(
        event.direction == StreamDirection.OUTGOING and event.message.get("method") == "initialize" for event in events
    )
