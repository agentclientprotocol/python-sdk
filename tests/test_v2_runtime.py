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

    async def session_update(
        self,
        session_id: str,
        update: v2.schema.UserMessageChunk
        | v2.schema.UserMessageUpdate
        | v2.schema.AgentMessageChunk
        | v2.schema.AgentMessageUpdate
        | v2.schema.AgentThoughtChunk
        | v2.schema.AgentThoughtUpdate
        | v2.schema.ToolCallContentChunkUpdate
        | v2.schema.SessionToolCallUpdate
        | v2.schema.SessionTerminalUpdate
        | v2.schema.SessionTerminalOutputChunk
        | v2.schema.SessionPlanUpdate
        | v2.schema.SessionPlanRemovedUpdate
        | v2.schema.AvailableCommandsUpdate
        | v2.schema.ConfigOptionUpdate
        | v2.schema.SessionInfoUpdate
        | v2.schema.UsageUpdate
        | v2.schema.SessionNotice
        | v2.schema.SessionCompactionUpdate
        | v2.schema.SessionCompactionSummaryChunk
        | v2.schema.OtherSessionUpdate
        | v2.schema.RunningSessionStateUpdate
        | v2.schema.IdleSessionStateUpdate
        | v2.schema.RequiresActionSessionStateUpdate
        | v2.schema.OtherSessionStateUpdate,
        **kwargs: Any,
    ) -> None:
        notification = v2.schema.UpdateSessionNotification(session_id=session_id, update=update, **kwargs)
        await self.updates.put(notification)


class Agent:
    def __init__(self, *, response_version: int = v2.PROTOCOL_VERSION) -> None:
        self.response_version = response_version
        self.initialize_calls = 0
        self.client_name: str | None = None

    async def initialize(
        self, protocol_version: int, info: v2.schema.Implementation, **kwargs: Any
    ) -> v2.schema.InitializeResponse:
        self.initialize_calls += 1
        self.client_name = info.name
        return v2.schema.InitializeResponse(
            protocol_version=self.response_version,
            info=v2.schema.Implementation(name="test-agent", version="1.0.0"),
            capabilities=v2.schema.AgentCapabilities(session=v2.schema.SessionCapabilities()),
        )

    async def new_session(self, cwd: str, **kwargs: Any) -> v2.schema.NewSessionResponse:
        request = v2.schema.NewSessionRequest(cwd=cwd, **kwargs)
        return v2.schema.NewSessionResponse(session_id=f"session:{request.cwd}")


class CallableAgent(Agent):
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("an agent object must not be treated as a factory")


class SessionAgent(Agent):
    def __init__(self, *, echo_before_response: bool) -> None:
        super().__init__()
        self.echo_before_response = echo_before_response

    def on_connect(self, connection: v2.AgentSideConnection) -> None:
        self.connection = connection

    async def new_session(self, cwd: str, **kwargs: Any) -> v2.schema.NewSessionResponse:
        request = v2.schema.NewSessionRequest(cwd=cwd, **kwargs)
        response = await super().new_session(cwd=request.cwd)
        await self.connection.session_update(session_id=response.session_id, update=v2.schema.IdleSessionStateUpdate())
        return response

    async def prompt(
        self,
        session_id: str,
        prompt: list[
            v2.schema.TextContentBlock
            | v2.schema.ImageContentBlock
            | v2.schema.AudioContentBlock
            | v2.schema.ResourceContentBlock
            | v2.schema.EmbeddedResourceContentBlock
            | v2.schema.OtherContentBlock
        ],
        **kwargs: Any,
    ) -> v2.schema.PromptResponse:
        request = v2.schema.PromptRequest(session_id=session_id, prompt=prompt, **kwargs)
        await self.connection.session_update(
            session_id=request.session_id, update=v2.schema.RunningSessionStateUpdate()
        )
        if self.echo_before_response:
            await self.echo_prompt(request)
        return v2.schema.PromptResponse(message_id="user-message-1")

    async def echo_prompt(self, request: v2.schema.PromptRequest) -> None:
        await self.connection.session_update(
            session_id=request.session_id,
            update=v2.schema.UserMessageUpdate(message_id="user-message-1", content=request.prompt),
        )


class ExtensionClient:
    async def handle_extension_request(self, method: str, params: Any) -> Any:
        return {"method": method, "params": params}


class ExtensionAgent:
    def __init__(self) -> None:
        self.notifications: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def initialize(
        self, protocol_version: int, info: v2.schema.Implementation, **kwargs: Any
    ) -> v2.schema.InitializeResponse:
        return v2.schema.InitializeResponse(
            protocol_version=v2.PROTOCOL_VERSION,
            info=v2.schema.Implementation(name="extension-agent", version="1.0.0"),
        )

    async def handle_extension_request(self, method: str, params: Any) -> Any:
        return {"method": method, "params": params}

    async def cancel_session(self, session_id: str, **kwargs: Any) -> None:
        notification = v2.schema.CancelSessionNotification(session_id=session_id, **kwargs)
        await self.notifications.put(("cancel", notification))

    async def notify_mcp(self, connection_id: str, method: str, **kwargs: Any) -> None:
        notification = v2.schema.MessageMcpNotification(connection_id=connection_id, method=method, **kwargs)
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
        initialized = await client_connection.initialize(
            protocol_version=v2.PROTOCOL_VERSION, info=v2.schema.Implementation(name="test-client", version="1.0.0")
        )
        session = await client_connection.new_session(cwd="/workspace")

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
            await client_connection.new_session(cwd="/workspace")
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
        initialized = await client_connection.initialize(
            protocol_version=v2.PROTOCOL_VERSION, info=v2.schema.Implementation(name="test-client", version="1.0.0")
        )

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
            await client_connection.initialize(
                protocol_version=1, info=v2.schema.Implementation(name="test-client", version="1.0.0")
            )

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
            await client_connection.initialize(
                protocol_version=v2.PROTOCOL_VERSION, info=v2.schema.Implementation(name="test-client", version="1.0.0")
            )

        assert isinstance(error.value, RequestError)
        assert error.value.code == -32600
    finally:
        await client_connection.close()
        await agent_connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("echo_before_response", [True, False])
async def test_session_updates_are_delivered_independently_from_prompt(echo_before_response: bool) -> None:
    client_transport, agent_transport = memory_transport_pair()
    client = SessionClient()
    agent = SessionAgent(echo_before_response=echo_before_response)
    agent_connection = v2.AgentSideConnection(agent, agent_transport)
    client_connection = v2.ClientSideConnection(client, client_transport)

    try:
        await client_connection.initialize(
            protocol_version=v2.PROTOCOL_VERSION, info=v2.schema.Implementation(name="test-client", version="1.0.0")
        )
        session = await client_connection.new_session(cwd="/workspace")
        request = v2.schema.PromptRequest(
            session_id=session.session_id,
            prompt=[v2.schema.TextContentBlock(text="hello")],
        )
        response = await client_connection.prompt(session_id=request.session_id, prompt=request.prompt)
        if not echo_before_response:
            await agent.echo_prompt(request)
        # Completion is independent traffic, sent after prompt acceptance.
        await agent_connection.session_update(
            session_id=session.session_id, update=v2.schema.IdleSessionStateUpdate(stop_reason="end_turn")
        )

        ready = await asyncio.wait_for(client.updates.get(), timeout=1)
        running = await asyncio.wait_for(client.updates.get(), timeout=1)
        echoed = await asyncio.wait_for(client.updates.get(), timeout=1)
        stopped = await asyncio.wait_for(client.updates.get(), timeout=1)

        assert isinstance(ready.update, v2.schema.IdleSessionStateUpdate)
        assert isinstance(running.update, v2.schema.RunningSessionStateUpdate)
        assert isinstance(echoed.update, v2.schema.UserMessageUpdate)
        assert echoed.update.message_id == response.message_id == "user-message-1"
        assert echoed.update.content == request.prompt
        assert isinstance(stopped.update, v2.schema.IdleSessionStateUpdate)
        assert stopped.update.stop_reason == "end_turn"
    finally:
        await client_connection.close()
        await agent_connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "update",
    [
        v2.schema.SessionNotice(
            severity="warning",
            title="Context is nearly full",
            description="Start a new session soon.",
            field_meta={"source": "context-monitor"},
        ),
        v2.schema.SessionCompactionUpdate(compaction_id="compact-1", status="in_progress"),
        v2.schema.SessionCompactionSummaryChunk(
            compaction_id="compact-1",
            content=v2.schema.TextContentBlock(text="Summary"),
        ),
    ],
)
async def test_notice_and_compaction_updates_reach_client(update) -> None:
    client_transport, agent_transport = memory_transport_pair()
    client = SessionClient()
    async with (
        v2.AgentSideConnection(Agent(), agent_transport) as agent_connection,
        v2.ClientSideConnection(client, client_transport) as client_connection,
    ):
        await client_connection.initialize(
            protocol_version=v2.PROTOCOL_VERSION, info=v2.schema.Implementation(name="test-client", version="1.0.0")
        )
        await agent_connection.session_update(session_id="session-1", update=update)
        received = await asyncio.wait_for(client.updates.get(), timeout=1)
        assert received.session_id == "session-1"
        assert received.update == update


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates",
    [
        [
            v2.schema.SessionToolCallUpdate(tool_call_id="tool-1", name="read_file"),
            v2.schema.SessionToolCallUpdate(tool_call_id="tool-1"),
            v2.schema.SessionToolCallUpdate(tool_call_id="tool-1", name=None, field_meta=None),
        ],
        [
            v2.schema.SessionTerminalUpdate(terminal_id="term-1", command="ls"),
            v2.schema.SessionTerminalUpdate(terminal_id="term-1"),
            v2.schema.SessionTerminalUpdate(terminal_id="term-1", command=None, field_meta=None),
        ],
    ],
)
async def test_session_patches_preserve_omitted_and_cleared_fields(updates) -> None:
    client_transport, agent_transport = memory_transport_pair()
    client = SessionClient()
    async with (
        v2.AgentSideConnection(Agent(), agent_transport) as agent_connection,
        v2.ClientSideConnection(client, client_transport) as client_connection,
    ):
        await client_connection.initialize(
            protocol_version=v2.PROTOCOL_VERSION, info=v2.schema.Implementation(name="test-client", version="1.0.0")
        )
        for update in updates:
            await agent_connection.session_update(session_id="session-1", update=update)
            received = await asyncio.wait_for(client.updates.get(), timeout=1)
            assert received.update.model_dump(by_alias=True, exclude_unset=True) == update.model_dump(
                by_alias=True, exclude_unset=True
            )


def test_v2_public_entry_point_is_explicit() -> None:
    exported: dict[str, Any] = {name: getattr(v2, name) for name in v2.__all__}

    assert exported["PROTOCOL_VERSION"] == 2
    assert exported["schema"] is v2.schema
    assert set(exported) == {
        "Agent",
        "Client",
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
        await client_connection.initialize(
            protocol_version=v2.PROTOCOL_VERSION, info=v2.schema.Implementation(name="test-client", version="1.0.0")
        )

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

        await client_connection.cancel_session(session_id="session-1")
        await client_connection.notify_mcp(connection_id="mcp-1", method="notifications/progress")

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
        await client_connection.initialize(
            protocol_version=v2.PROTOCOL_VERSION, info=v2.schema.Implementation(name="test-client", version="1.0.0")
        )
        await agent_connection.session_update(session_id="session-1", update=v2.schema.IdleSessionStateUpdate())
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
        await client_connection.initialize(
            protocol_version=v2.PROTOCOL_VERSION, info=v2.schema.Implementation(name="test-client", version="1.0.0")
        )
        with pytest.raises(RequestError) as error:
            await client_connection.new_session(cwd="/workspace")

        assert isinstance(error.value, RequestError)
        assert error.value.code == -32601
    finally:
        await client_connection.close()
        await agent_connection.close()


@pytest.mark.asyncio
async def test_expanded_union_calls_round_trip_with_metadata() -> None:
    configurations: list[tuple[Any, Any, Any]] = []
    elicitations: list[dict[str, Any]] = []

    class ConfigAgent(Agent):
        async def set_config_option(self, config_id, session_id, value, *, type, **kwargs):  # noqa: A002
            assert (config_id, session_id, kwargs) == ("option", "s", {"trace": "config"})
            configurations.append((type, value, kwargs))
            return v2.schema.SetSessionConfigOptionResponse(config_options=[])

    class ElicitationClient:
        async def create_elicitation(self, message, mode, **kwargs):
            assert message == "Input"
            elicitations.append({"mode": mode, **kwargs})
            return v2.schema.DeclineElicitationResponse()

    client_transport, agent_transport = memory_transport_pair()
    async with (
        v2.AgentSideConnection(ConfigAgent(), agent_transport) as agent_connection,
        v2.ClientSideConnection(ElicitationClient(), client_transport) as client_connection,
    ):
        await client_connection.initialize(
            protocol_version=v2.PROTOCOL_VERSION,
            info=v2.schema.Implementation(name="test", version="1"),
        )
        for value, tag in [(True, None), ("fast", None), (10, "vendor/number")]:
            await client_connection.set_config_option("option", "s", value, type=tag, trace="config")
        assert [(tag, value) for tag, value, _ in configurations] == [
            ("boolean", True),
            ("id", "fast"),
            ("vendor/number", 10),
        ]
        await agent_connection.create_elicitation(
            "Input",
            "form",
            session_id="s",
            tool_call_id="tool",
            requested_schema=v2.schema.ElicitationSchema(properties={}),
            trace="form",
        )
        await agent_connection.create_elicitation(
            "Input",
            "url",
            request_id=7,
            elicitation_id="e",
            url="https://example.com/",
            trace="url",
        )
        await agent_connection.create_elicitation("Input", "vendor/custom", request_id=None)
        assert elicitations[0]["session_id"] == "s"
        assert elicitations[0]["tool_call_id"] == "tool"
        assert elicitations[0]["trace"] == "form"
        assert isinstance(elicitations[0]["requested_schema"], v2.schema.ElicitationSchema)
        assert elicitations[1]["request_id"] == 7
        assert elicitations[1]["elicitation_id"] == "e"
        assert elicitations[1]["trace"] == "url"
        assert elicitations[2]["request_id"] is None
        with pytest.raises(ValueError, match="either session_id or request_id"):
            await agent_connection.create_elicitation("Input", "vendor/custom", session_id="s", request_id=7)
