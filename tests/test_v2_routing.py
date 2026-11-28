from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from acp._transport import memory_transport_pair
from acp.connection import Connection
from acp.exceptions import RequestError
from acp.experimental import v2

PEER_INFO = v2.schema.Implementation(name="peer", version="1")


class _Agent(v2.Agent):
    async def initialize(self, protocol_version: int, info: v2.schema.Implementation, **kwargs: Any):
        return v2.schema.InitializeResponse(protocol_version=protocol_version, info=PEER_INFO)


async def _initialize_peer(method: str, params: Any, is_notification: bool) -> dict[str, Any]:
    assert method == "initialize"
    assert not is_notification
    return {"protocolVersion": params["protocolVersion"], "info": {"name": "peer", "version": "1"}}


@asynccontextmanager
async def _agent_peer(agent: _Agent) -> AsyncIterator[Connection]:
    peer_transport, agent_transport = memory_transport_pair()
    async with (
        v2.AgentSideConnection(agent, agent_transport),
        Connection(_initialize_peer, peer_transport) as peer,
    ):
        await asyncio.wait_for(
            peer.send_request("initialize", {"protocolVersion": v2.PROTOCOL_VERSION, "info": PEER_INFO.model_dump()}),
            timeout=2,
        )
        yield peer


@pytest.mark.asyncio
async def test_protocol_stubs_are_not_treated_as_implemented_handlers(caplog: pytest.LogCaptureFixture) -> None:
    async with _agent_peer(_Agent()) as peer:
        await peer.send_notification("session/cancel", {})
        with pytest.raises(RequestError, check=lambda error: error.code == -32601):
            await asyncio.wait_for(peer.send_request("session/new", {"cwd": "/workspace"}), timeout=2)
    assert not caplog.records


@pytest.mark.asyncio
async def test_v2_route_validation_and_empty_responses_remain_strict() -> None:
    class Agent(_Agent):
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
    async with _agent_peer(agent) as peer:
        for prompt in ([{"type": "text"}], []):
            with pytest.raises(RequestError, check=lambda error: error.code == -32602):
                await asyncio.wait_for(peer.send_request("session/prompt", {"sessionId": "s", "prompt": prompt}), 2)
            assert agent.calls == 0
        with pytest.raises(RequestError, check=lambda error: error.code == -32603):
            await asyncio.wait_for(
                peer.send_request("session/prompt", {"sessionId": "s", "prompt": [{"type": "text", "text": "hi"}]}), 2
            )
        assert agent.calls == 1
        assert await asyncio.wait_for(peer.send_request("auth/logout", {}), 2) == {}
        with pytest.raises(RequestError, check=lambda error: error.code == -32601):
            await asyncio.wait_for(
                peer.send_request("mcp/message", {"serverId": "s", "requestId": "r", "method": "ping"}), 2
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tag", "value"),
    [("id", "fast"), ("boolean", True), ("vendor/number", 10)],
)
async def test_config_union_flattens_selected_branch_and_metadata(tag, value) -> None:
    class Agent(_Agent):
        async def set_config_option(self, config_id, session_id, value, *, type, **kwargs):  # noqa: A002
            assert (config_id, session_id, type) == ("option", "s", tag)
            assert kwargs == {"trace": "t"}
            return {"configOptions": []}

    async with _agent_peer(Agent()) as peer:
        params = {
            "configId": "option",
            "sessionId": "s",
            "type": tag,
            "value": value,
            "_meta": {"trace": "t"},
        }
        result = await asyncio.wait_for(peer.send_request("session/set_config_option", params), 2)
        assert result == {"configOptions": []}
        with pytest.raises(RequestError, check=lambda error: error.code == -32602):
            await asyncio.wait_for(
                peer.send_request("session/set_config_option", {**params, "type": "boolean", "value": {"bad": True}}), 2
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [{"sessionId": "s", "toolCallId": "tool"}, {"requestId": 7}, {"requestId": "rpc-7"}])
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
    peer_transport, client_transport = memory_transport_pair()
    async with (
        v2.ClientSideConnection(Client(), client_transport) as client,
        Connection(_initialize_peer, peer_transport) as peer,
    ):
        await asyncio.wait_for(client.initialize(v2.PROTOCOL_VERSION, PEER_INFO), 2)
        result = await asyncio.wait_for(peer.send_request("elicitation/create", params), 2)
    assert result == {"action": "decline"}
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
