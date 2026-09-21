"""Exercise v1 methods added after the original session/terminal API."""

import asyncio
import inspect
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from acp import schema
from acp.agent.connection import AgentSideConnection
from acp.agent.router import build_agent_router
from acp.client.connection import ClientSideConnection
from acp.client.router import build_client_router
from acp.exceptions import RequestError
from acp.interfaces import Agent, Client
from acp.meta import AGENT_METHODS, CLIENT_METHODS

REQUESTS = [
    ("delete_session", "session/delete", {"session_id": "s"}, schema.DeleteSessionResponse()),
    ("list_providers", "providers/list", {}, schema.ListProvidersResponse(providers=[])),
    (
        "set_provider",
        "providers/set",
        {"provider_id": "p", "api_type": "custom", "base_url": "https://example.com", "headers": {"X-Test": "test"}},
        schema.SetProviderResponse(),
    ),
    ("disable_provider", "providers/disable", {"provider_id": "p"}, schema.DisableProviderResponse()),
    ("logout", "logout", {}, schema.LogoutResponse()),
    ("start_nes", "nes/start", {"workspace_uri": "file:///workspace"}, schema.StartNesResponse(session_id="n")),
    (
        "suggest_nes",
        "nes/suggest",
        {
            "session_id": "n",
            "uri": "file:///a",
            "version": 1,
            "position": schema.Position(line=0, character=0),
            "trigger_kind": "manual",
        },
        schema.SuggestNesResponse(suggestions=[]),
    ),
    ("close_nes", "nes/close", {"session_id": "n"}, schema.CloseNesResponse()),
]
POSITION = schema.Position(line=0, character=0)
NOTIFICATIONS = [
    ("accept_nes", "nes/accept", {"session_id": "n", "id": "edit"}),
    ("reject_nes", "nes/reject", {"session_id": "n", "id": "edit", "reason": "rejected"}),
    (
        "did_open",
        "document/didOpen",
        {"session_id": "n", "uri": "file:///a", "language_id": "python", "version": 1, "text": "hello"},
    ),
    (
        "did_change",
        "document/didChange",
        {
            "session_id": "n",
            "uri": "file:///a",
            "version": 2,
            "content_changes": [schema.TextDocumentContentChangeEvent(text="world")],
        },
    ),
    ("did_close", "document/didClose", {"session_id": "n", "uri": "file:///a"}),
    ("did_save", "document/didSave", {"session_id": "n", "uri": "file:///a", "text": "world"}),
    (
        "did_focus",
        "document/didFocus",
        {
            "session_id": "n",
            "uri": "file:///a",
            "version": 2,
            "position": POSITION,
            "visible_range": schema.Range(start=POSITION, end=POSITION),
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "wire_method", "params", "response"), REQUESTS)
async def test_agent_request_roundtrip(connect, agent, name, wire_method, params, response):
    handler = AsyncMock(return_value=response)
    setattr(agent, name, handler)
    _, connection = connect(use_unstable_protocol=True)
    result = await getattr(connection, name)(**params, trace="test")
    assert result == response
    assert handler.await_args is not None
    assert handler.await_args.kwargs.items() >= (params | {"trace": "test"}).items()


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "wire_method", "params"), NOTIFICATIONS)
async def test_agent_notification_roundtrip(connect, agent, name, wire_method, params):
    received = asyncio.Event()
    calls = []

    async def handler(**kwargs):
        calls.append(kwargs)
        received.set()

    setattr(agent, name, handler)
    _, connection = connect(use_unstable_protocol=True)
    await getattr(connection, name)(**params, trace="test")
    await asyncio.wait_for(received.wait(), 2)
    assert calls[0].items() >= (params | {"trace": "test"}).items()


@pytest.mark.asyncio
async def test_mcp_connection_lifecycle(connect, client):
    client.connect_mcp = AsyncMock(return_value=schema.ConnectMcpResponse(connection_id="c"))
    client.disconnect_mcp = AsyncMock(return_value=None)
    connection, _ = connect(use_unstable_protocol=True)
    assert (await connection.connect_mcp(server_id="m", trace="test")).connection_id == "c"
    client.connect_mcp.assert_awaited_once_with(server_id="m", trace="test")
    assert await connection.disconnect_mcp(connection_id="c") == schema.DisconnectMcpResponse()
    client.disconnect_mcp.assert_awaited_once_with(connection_id="c")


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, {}, {"tools": []}, [1, "x"], False, 42])
@pytest.mark.parametrize("direction", ["agent", "client"])
async def test_mcp_requests_and_notifications_share_method(connect, agent, client, direction, response):
    receiver = agent if direction == "agent" else client
    receiver.mcp_message = AsyncMock(return_value=response)
    received = asyncio.Event()

    async def notify_mcp(**kwargs):
        assert kwargs == {"connection_id": "c", "method": "notifications/initialized", "params": None, "trace": "test"}
        received.set()

    receiver.notify_mcp = notify_mcp
    agent_side, client_side = connect(use_unstable_protocol=True)
    sender = client_side if direction == "agent" else agent_side
    assert (
        await sender.mcp_message(connection_id="c", method="tools/list", params={"cursor": "next"}, trace="test")
        == response
    )
    receiver.mcp_message.assert_awaited_once_with(
        connection_id="c", method="tools/list", params={"cursor": "next"}, trace="test"
    )
    await sender.notify_mcp(connection_id="c", method="notifications/initialized", trace="test")
    await asyncio.wait_for(received.wait(), 2)


@pytest.mark.asyncio
async def test_delete_session_empty_and_legacy_response(connect, agent):
    agent.delete_session = AsyncMock(return_value=None)
    _, connection = connect(use_unstable_protocol=True)
    with pytest.warns(DeprecationWarning):
        result = await connection.deleteSession(schema.DeleteSessionRequest(session_id="s"))
    assert result == schema.DeleteSessionResponse()
    agent.delete_session.assert_awaited_once_with(session_id="s")


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "wire_method", "params", "response"), REQUESTS)
async def test_unimplemented_requests_report_method_not_found(name, wire_method, params, response):
    router = build_agent_router(cast(Agent, object()), use_unstable_protocol=True)
    with pytest.raises(RequestError) as exc:
        await router(wire_method, {}, False)
    assert isinstance(exc.value, RequestError)
    assert exc.value.code == -32601


@pytest.mark.asyncio
async def test_delete_session_validation():
    class Handler:
        async def delete_session(self, session_id: str, **kwargs: Any):
            return None

    router = build_agent_router(cast(Agent, Handler()))
    assert await router("session/delete", {"sessionId": "s"}, False) == {}
    with pytest.raises(ValidationError):
        await router("session/delete", {}, False)


@pytest.mark.parametrize(
    ("builder", "methods", "connection", "interface"),
    [
        (build_agent_router, AGENT_METHODS, ClientSideConnection, Agent),
        (build_client_router, CLIENT_METHODS, AgentSideConnection, Client),
    ],
)
def test_all_schema_methods_have_routes_and_senders(builder, methods, connection, interface):
    router = builder(object(), use_unstable_protocol=True)
    assert set(router._requests) | set(router._notifications) == set(methods.values())
    source = inspect.getsource(connection)
    for key in methods:
        assert f'["{key}"]' in source
    for name, method in inspect.getmembers(connection, inspect.isfunction):
        if hasattr(method, "__param_model__"):
            assert hasattr(interface, name) or any(char.isupper() for char in name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "params", "response"),
    [
        ("delete_session", {"session_id": "s"}, schema.DeleteSessionResponse()),
        ("logout", {}, schema.LogoutResponse()),
    ],
)
async def test_stable_methods_work_without_unstable_flag(connect, agent, name, params, response):
    handler = AsyncMock(return_value=response)
    setattr(agent, name, handler)
    _, connection = connect()
    assert await getattr(connection, name)(**params) == response
    handler.assert_awaited_once_with(**params)
