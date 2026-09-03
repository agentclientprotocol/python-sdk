from __future__ import annotations

import asyncio
from typing import Any

import pytest

from acp._transport import memory_transport_pair
from acp.exceptions import RequestError
from acp.experimental import v2


class Client:
    pass


class SessionClient:
    def __init__(self) -> None:
        self.updates: asyncio.Queue[v2.schema.UpdateSessionNotification] = asyncio.Queue()

    async def session_update(self, notification: v2.schema.UpdateSessionNotification) -> None:
        await self.updates.put(notification)


class Agent:
    def __init__(self, *, response_version: int = v2.PROTOCOL_VERSION) -> None:
        self.response_version = response_version
        self.initialize_calls = 0
        self.client_name: str | None = None

    async def initialize(self, request: v2.schema.InitializeRequest) -> v2.schema.InitializeResponse:
        self.initialize_calls += 1
        self.client_name = request.info.name
        return v2.schema.InitializeResponse(
            protocol_version=self.response_version,
            info=v2.schema.Implementation(name="test-agent", version="1.0.0"),
            capabilities=v2.schema.AgentCapabilities(session=v2.schema.SessionCapabilities()),
        )

    async def new_session(self, request: v2.schema.NewSessionRequest) -> v2.schema.NewSessionResponse:
        return v2.schema.NewSessionResponse(session_id=f"session:{request.cwd}")


class CallableAgent(Agent):
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("an agent object must not be treated as a factory")


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


class ExtensionClient:
    async def handle_extension_request(self, method: str, params: Any) -> Any:
        return {"method": method, "params": params}


class ExtensionAgent:
    def __init__(self) -> None:
        self.notifications: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def initialize(self, request: v2.schema.InitializeRequest) -> v2.schema.InitializeResponse:
        return v2.schema.InitializeResponse(
            protocol_version=v2.PROTOCOL_VERSION,
            info=v2.schema.Implementation(name="extension-agent", version="1.0.0"),
        )

    async def handle_extension_request(self, method: str, params: Any) -> Any:
        return {"method": method, "params": params}

    async def cancel_session(self, notification: v2.schema.CancelSessionNotification) -> None:
        await self.notifications.put(("cancel", notification))

    async def notify_mcp(self, notification: v2.schema.MessageMcpNotification) -> None:
        await self.notifications.put(("mcp", notification))


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
        assert agent.client_name == "test-client"
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
async def test_callable_agent_is_not_treated_as_a_factory() -> None:
    client_transport, agent_transport = memory_transport_pair()
    agent = CallableAgent()
    agent_connection = v2.AgentSideConnection(agent, agent_transport)
    client_connection = v2.ClientSideConnection(Client(), client_transport)

    try:
        initialized = await client_connection.initialize(initialize_request())

        assert initialized.protocol_version == v2.PROTOCOL_VERSION
        assert agent.initialize_calls == 1
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
async def test_session_updates_are_delivered_independently_from_prompt() -> None:
    client_transport, agent_transport = memory_transport_pair()
    client = SessionClient()
    agent_connection = v2.AgentSideConnection(SessionAgent(), agent_transport)
    client_connection = v2.ClientSideConnection(client, client_transport)

    try:
        await client_connection.initialize(initialize_request())
        session = await client_connection.new_session(v2.schema.NewSessionRequest(cwd="/workspace"))
        await client_connection.prompt(
            v2.schema.PromptRequest(
                session_id=session.session_id,
                prompt=[v2.schema.TextContentBlock(text="hello")],
            )
        )

        ready = await asyncio.wait_for(client.updates.get(), timeout=1)
        running = await asyncio.wait_for(client.updates.get(), timeout=1)
        stopped = await asyncio.wait_for(client.updates.get(), timeout=1)

        assert isinstance(ready.update, v2.schema.IdleSessionStateUpdate)
        assert isinstance(running.update, v2.schema.RunningSessionStateUpdate)
        assert isinstance(stopped.update, v2.schema.IdleSessionStateUpdate)
        assert stopped.update.stop_reason == "end_turn"
    finally:
        await client_connection.close()
        await agent_connection.close()


def test_v2_public_entry_point_is_explicit() -> None:
    exported: dict[str, Any] = {name: getattr(v2, name) for name in v2.__all__}

    assert exported["PROTOCOL_VERSION"] == 2
    assert exported["schema"] is v2.schema
    assert set(exported) == {
        "AgentSideConnection",
        "ClientSideConnection",
        "PROTOCOL_VERSION",
        "connect_to_agent",
        "run_agent",
        "schema",
    }
    assert "InitializeRequest" not in v2.__all__


@pytest.mark.asyncio
async def test_extension_and_notification_names_are_explicit() -> None:
    client_transport, agent_transport = memory_transport_pair()
    agent = ExtensionAgent()
    agent_connection = v2.AgentSideConnection(agent, agent_transport)
    client_connection = v2.ClientSideConnection(ExtensionClient(), client_transport)

    try:
        await client_connection.initialize(initialize_request())

        assert await client_connection.send_extension_request("_vendor/do", {"value": 1}) == {
            "method": "_vendor/do",
            "params": {"value": 1},
        }
        assert await agent_connection.send_extension_request("_vendor/read", {"value": 2}) == {
            "method": "_vendor/read",
            "params": {"value": 2},
        }
        with pytest.raises(ValueError, match="must start with '_'"):
            await client_connection.send_extension_request("vendor/do")

        await client_connection.cancel_session(v2.schema.CancelSessionNotification(session_id="session-1"))
        await client_connection.notify_mcp(
            v2.schema.MessageMcpNotification(connection_id="mcp-1", method="notifications/progress")
        )

        cancel_kind, cancel = await asyncio.wait_for(agent.notifications.get(), timeout=1)
        mcp_kind, mcp = await asyncio.wait_for(agent.notifications.get(), timeout=1)
        assert (cancel_kind, cancel.session_id) == ("cancel", "session-1")
        assert (mcp_kind, mcp.connection_id) == ("mcp", "mcp-1")
    finally:
        await client_connection.close()
        await agent_connection.close()


@pytest.mark.asyncio
async def test_unhandled_notifications_are_ignored(caplog: pytest.LogCaptureFixture) -> None:
    client_transport, agent_transport = memory_transport_pair()
    agent_connection = v2.AgentSideConnection(Agent(), agent_transport)
    client_connection = v2.ClientSideConnection(object(), client_transport)

    try:
        await client_connection.initialize(initialize_request())
        await agent_connection.session_update(
            v2.schema.UpdateSessionNotification(
                session_id="session-1",
                update=v2.schema.IdleSessionStateUpdate(),
            )
        )
        await client_connection.send_extension_notification("_vendor/event")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert not [record for record in caplog.records if record.levelno >= 40]
    finally:
        await client_connection.close()
        await agent_connection.close()


@pytest.mark.asyncio
async def test_missing_request_handler_returns_method_not_found() -> None:
    client_transport, agent_transport = memory_transport_pair()
    agent_connection = v2.AgentSideConnection(ExtensionAgent(), agent_transport)
    client_connection = v2.ClientSideConnection(object(), client_transport)

    try:
        await client_connection.initialize(initialize_request())
        with pytest.raises(RequestError) as error:
            await client_connection.new_session(v2.schema.NewSessionRequest(cwd="/workspace"))

        assert isinstance(error.value, RequestError)
        assert error.value.code == -32601
    finally:
        await client_connection.close()
        await agent_connection.close()
