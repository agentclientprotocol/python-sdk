from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic import ValidationError

from acp.exceptions import RequestError
from acp.experimental import v2
from acp.experimental.v2._methods import protocol_specs
from acp.experimental.v2._router import MethodRouter
from acp.experimental.v2.meta import AGENT_METHODS, CLIENT_METHODS


@pytest.mark.parametrize(
    ("protocol", "connection", "methods"),
    [(v2.Agent, v2.ClientSideConnection, AGENT_METHODS), (v2.Client, v2.AgentSideConnection, CLIENT_METHODS)],
)
def test_protocols_cover_wire_methods_and_match_connection_signatures(protocol, connection, methods) -> None:
    requests, notifications = protocol_specs(protocol)
    assert {spec.method for spec in (*requests, *notifications)} == set(methods.values())
    for spec in (*requests, *notifications):
        assert inspect.signature(getattr(protocol, spec.handler)) == inspect.signature(
            getattr(connection, spec.handler)
        )
    assert "mcp/message" in {spec.method for spec in requests}
    assert "mcp/message" in {spec.method for spec in notifications}


@pytest.mark.asyncio
async def test_protocol_stubs_are_not_treated_as_implemented_handlers() -> None:
    class Agent(v2.Agent):
        pass

    router = MethodRouter(Agent(), v2.Agent)
    with pytest.raises(RequestError) as error:
        await router("session/new", {"cwd": "/workspace"}, False)
    assert isinstance(error.value, RequestError)
    assert error.value.code == -32601
    assert await router("session/cancel", {"sessionId": "s"}, True) is None


@pytest.mark.asyncio
async def test_v2_route_validation_and_empty_responses_remain_strict() -> None:
    class Agent:
        def __init__(self) -> None:
            self.calls = 0

        async def prompt(self, **kwargs: Any) -> Any:
            self.calls += 1
            return {"stopReason": "end_turn"}  # v1 response must not pass v2 validation.

        async def logout(self, **kwargs: Any) -> None:
            pass

        async def mcp_message(self, **kwargs: Any) -> Any:
            return None

    agent = Agent()
    router = MethodRouter(agent, v2.Agent)
    with pytest.raises(ValidationError):
        await router("session/prompt", {"sessionId": "s", "prompt": [{"type": "text"}]}, False)
    assert agent.calls == 0
    with pytest.raises(ValidationError):
        await router("session/prompt", {"sessionId": "s", "prompt": []}, False)
    assert agent.calls == 1
    assert isinstance(await router("auth/logout", {}, False), v2.schema.LogoutAuthResponse)
    assert await router("mcp/message", {"connectionId": "m", "method": "ping"}, False) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tag", "value"),
    [("id", "fast"), ("boolean", True), ("vendor/number", 10)],
)
async def test_config_union_flattens_selected_branch_and_metadata(tag, value) -> None:
    class Agent:
        async def set_config_option(self, config_id, session_id, value, *, type, **kwargs):  # noqa: A002
            assert (config_id, session_id, type) == ("option", "s", tag)
            assert kwargs == {"trace": "t"}
            return {"configOptions": []}

    router = MethodRouter(Agent(), v2.Agent)
    result = await router(
        "session/set_config_option",
        {
            "configId": "option",
            "sessionId": "s",
            "type": tag,
            "value": value,
            "_meta": {"trace": "t"},
        },
        False,
    )
    assert isinstance(result, v2.schema.SetSessionConfigOptionResponse)
    with pytest.raises(ValidationError):
        await router(
            "session/set_config_option",
            {
                "configId": "option",
                "sessionId": "s",
                "type": "boolean",
                "value": {"bad": True},
            },
            False,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [{"sessionId": "s", "toolCallId": "tool"}, {"requestId": 7}, {"requestId": None}])
@pytest.mark.parametrize("mode", ["form", "url", "vendor/custom"])
async def test_elicitation_retains_branch_specific_fields(scope, mode) -> None:
    captured = {}

    class Client:
        async def create_elicitation(self, **kwargs):
            captured.update(kwargs)
            return {"action": "decline"}

    params = {"message": "Input", "mode": mode, **scope, "_meta": {"trace": "t"}}
    if mode == "form":
        params["requestedSchema"] = {"type": "object", "properties": {}}
    if mode == "url":
        params.update(elicitationId="e", url="https://example.com/")
    result = await MethodRouter(Client(), v2.Client)("elicitation/create", params, False)
    assert isinstance(result, v2.schema.DeclineElicitationResponse)
    assert captured["mode"] == mode
    assert captured["trace"] == "t"
    if "sessionId" in scope:
        assert captured["session_id"] == "s"
        assert captured["tool_call_id"] == "tool"
        assert "request_id" not in captured
    else:
        assert captured["request_id"] == scope["requestId"]
        assert "session_id" not in captured
    if mode == "form":
        assert isinstance(captured["requested_schema"], v2.schema.ElicitationSchema)
    if mode == "url":
        assert captured["elicitation_id"] == "e"
        assert str(captured["url"]) == "https://example.com/"
