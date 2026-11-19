"""Stateless MCP wire contracts and loopback behavior for both ACP versions."""

import asyncio
from typing import Any, cast

import pytest
import pytest_asyncio
from pydantic import TypeAdapter, ValidationError

from acp import schema as v1_schema
from acp._transport import memory_transport_pair
from acp.agent.connection import AgentSideConnection
from acp.client.connection import ClientSideConnection
from acp.exceptions import RequestError
from acp.experimental import v2
from acp.interfaces import Agent as AgentProtocol
from acp.interfaces import Client as ClientProtocol


@pytest.fixture(params=[v1_schema, v2.schema], ids=["v1", "v2"])
def schema(request):
    return request.param


@pytest.mark.parametrize(
    "result", [None, False, 0, "text", [None], {"resultType": "input_required", "requestState": {"opaque": None}}]
)
def test_result_carrier_preserves_opaque_json_and_prefers_result(schema, result):
    carrier = TypeAdapter(schema.MessageMcpResponse).validate_python({
        "result": result,
        "error": {"code": -32000, "message": "ignored"},
        "unknown": True,
        "_meta": "invalid",
    })
    assert isinstance(carrier, schema.McpResultResponse)
    assert carrier.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True) == {"result": result}


@pytest.mark.parametrize("data", [{}, {"data": None}, {"data": {"nested": None}}])
def test_mcp_error_preserves_data_presence_and_extensions(schema, data):
    error = {"code": -32000, "message": "MCP error, not ACP auth", "vendor": {"extra": None}, "nullable": None, **data}
    carrier = TypeAdapter(schema.MessageMcpResponse).validate_python({"error": error, "_meta": {"trace": "outer"}})
    assert isinstance(carrier, schema.McpErrorResponse)
    assert carrier.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True) == {
        "error": error,
        "_meta": {"trace": "outer"},
    }


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"error": None},
        {"error": {}},
        {"error": {"code": True, "message": "bad"}},
        {"error": {"code": "1", "message": "bad"}},
    ],
)
def test_invalid_carriers_are_rejected(schema, value):
    with pytest.raises(ValidationError):
        TypeAdapter(schema.MessageMcpResponse).validate_python(value)


@pytest.mark.parametrize("model_name", ["MessageMcpRequest", "MessageMcpNotification"])
@pytest.mark.parametrize(
    "invalid",
    [
        {"connectionId": "old", "method": "ping"},
        {"serverId": "s", "method": "ping"},
        {"serverId": "s", "requestId": 1, "method": "ping"},
        {"serverId": "s", "requestId": "r", "method": "ping", "params": []},
    ],
)
def test_envelope_requires_server_and_logical_request_ids(schema, model_name, invalid):
    with pytest.raises(ValidationError):
        getattr(schema, model_name).model_validate(invalid)


@pytest_asyncio.fixture
async def peers(schema):
    class Agent:
        def __init__(self):
            self.notifications = asyncio.Queue()

        async def initialize(self, **kwargs):
            if schema is v2.schema:
                return schema.InitializeResponse(
                    protocol_version=2, info=schema.Implementation(name="test", version="1")
                )
            return schema.InitializeResponse(protocol_version=1)

        async def notify_mcp(self, **kwargs):
            await self.notifications.put(kwargs)

    class Client:
        def __init__(self):
            self.outcome: Any = {"result": None}
            self.calls = []

        async def mcp_message(self, **kwargs):
            self.calls.append(kwargs)
            if isinstance(self.outcome, Exception):
                raise self.outcome
            if callable(self.outcome):
                return await self.outcome(**kwargs)
            return self.outcome

    agent, client = Agent(), Client()
    a_transport, c_transport = memory_transport_pair()
    if schema is v2.schema:
        a = v2.AgentSideConnection(cast(AgentProtocol, agent), a_transport)
        c = v2.ClientSideConnection(cast(ClientProtocol, client), c_transport)
        await c.initialize(protocol_version=2, info=schema.Implementation(name="test", version="1"))
    else:
        a = AgentSideConnection(cast(AgentProtocol, agent), a_transport, use_unstable_protocol=True)
        c = ClientSideConnection(cast(ClientProtocol, client), c_transport, use_unstable_protocol=True)
    try:
        yield a, c, agent, client
    finally:
        await c.close()
        await a.close()


@pytest.mark.asyncio
async def test_request_metadata_notifications_and_error_domains(peers, schema):
    a, c, agent, client = peers
    inner = {
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {},
            "progressToken": "p",
        }
    }

    async def handle(**kwargs):
        await c.notify_mcp(
            server_id=kwargs["server_id"],
            request_id=kwargs["request_id"],
            method="notifications/progress",
            params={"progressToken": "p", "progress": 1},
            trace="notification",
        )
        return {"error": {"code": -32000, "message": "inner", "data": None, "vendor": 1}}

    client.outcome = handle
    response = await a.mcp_message(
        server_id="s", request_id="logical", method="tools/call", params=inner, trace="outer"
    )
    assert isinstance(response, schema.McpErrorResponse)
    assert response.error.code == -32000
    assert response.error.model_dump(exclude_unset=True) == {
        "code": -32000,
        "message": "inner",
        "data": None,
        "vendor": 1,
    }
    assert client.calls == [
        {"server_id": "s", "request_id": "logical", "method": "tools/call", "params": inner, "trace": "outer"}
    ]
    notification = await asyncio.wait_for(agent.notifications.get(), 1)
    assert notification == {
        "server_id": "s",
        "request_id": "logical",
        "method": "notifications/progress",
        "params": {"progressToken": "p", "progress": 1},
        "trace": "notification",
    }
    client.outcome = RequestError(-33001, "server unavailable")
    with pytest.raises(RequestError) as error:
        await a.mcp_message(server_id="s", request_id="next", method="tools/list")
    assert isinstance(error.value, RequestError)
    assert error.value.code == -33001


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", [{"result": None}, {"result": {"resultType": "input_required", "requestState": {"opaque": None}}}]
)
async def test_loopback_preserves_null_and_mrtr_result(peers, outcome):
    a, _, _, client = peers
    client.outcome = outcome
    response = await a.mcp_message(server_id="s", request_id="r", method="tools/call")
    assert response.result == outcome["result"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", [None, {}, {"result": None, "error": {"code": 1, "message": "both"}}, {"error": None}]
)
async def test_invalid_outgoing_outcomes_are_binding_failures(peers, outcome):
    a, _, _, client = peers
    client.outcome = outcome
    with pytest.raises(RequestError) as error:
        await a.mcp_message(server_id="s", request_id="r", method="tools/call")
    assert isinstance(error.value, RequestError)
    assert error.value.code == -33002


@pytest.mark.asyncio
async def test_cancellation_is_per_outer_request(peers):
    a, _, _, client = peers
    started, cleaned = asyncio.Event(), asyncio.Event()

    async def handle(request_id, **kwargs):
        if request_id == "sibling":
            return {"result": None}
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    client.outcome = handle
    pending = asyncio.create_task(a.mcp_message(server_id="s", request_id="long", method="subscriptions/listen"))
    await asyncio.wait_for(started.wait(), 1)
    assert (await a.mcp_message(server_id="s", request_id="sibling", method="tools/list")).result is None
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await asyncio.wait_for(cleaned.wait(), 1)


@pytest.mark.asyncio
async def test_removed_binding_methods_and_reverse_calls_are_not_exposed(peers):
    a, c, _, _ = peers
    assert not hasattr(a, "connect_mcp")
    assert not hasattr(a, "disconnect_mcp")
    assert not hasattr(a, "notify_mcp")
    assert not hasattr(c, "mcp_message")
    for method in ("mcp/connect", "mcp/disconnect"):
        with pytest.raises(RequestError) as error:
            await a._conn.send_request(method, {"serverId": "s"})
        assert isinstance(error.value, RequestError)
        assert error.value.code == -32601


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [{}, {"data": None}])
@pytest.mark.parametrize("extensions", [{}, {"vendor": None}])
async def test_loopback_preserves_error_data_presence(peers, data, extensions):
    a, _, _, client = peers
    error = {"code": -32602, "message": "inner", **data, **extensions}
    client.outcome = {"error": error}
    response = await a.mcp_message(server_id="s", request_id="r", method="tools/call")
    assert response.error.model_dump(exclude_unset=True) == error
