"""``$/cancel_request`` handling (https://agentclientprotocol.com/protocol/v1/cancellation)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, cast

import pytest

from acp import Agent
from acp.core import AgentSideConnection, ClientSideConnection
from acp.schema import PermissionOption, ToolCallUpdate
from tests.conftest import TestAgent, TestClient


async def _write(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    writer.write((json.dumps(message) + "\n").encode())
    await writer.drain()


async def _read(reader: asyncio.StreamReader) -> dict[str, Any]:
    return json.loads(await asyncio.wait_for(reader.readline(), timeout=1))


def _prompt(request_id: int | str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "session/prompt",
        "params": {"sessionId": "sess", "prompt": [{"type": "text", "text": "hi"}]},
    }


def _cancel_request(request_id: int | str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": "$/cancel_request", "params": {"requestId": request_id}}


class _BlockingAgent(TestAgent):
    """Blocks every prompt until cancelled and records how each one ended."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled: list[str] = []

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> Any:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.append(session_id)
            raise
        return await super().prompt(session_id, prompt, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_id", [7, "req-7"])
async def test_cancel_request_cancels_handler_and_replies_request_cancelled(
    server, caplog: pytest.LogCaptureFixture, request_id: int | str
) -> None:
    agent = _BlockingAgent()
    async with AgentSideConnection(cast(Agent, agent), server.server_writer, server.server_reader, listening=True):
        with caplog.at_level(logging.ERROR):
            await _write(server.client_writer, _prompt(request_id))
            await asyncio.wait_for(agent.started.wait(), timeout=1)
            await _write(server.client_writer, _cancel_request(request_id))
            response = await _read(server.client_reader)

        assert response["id"] == request_id
        assert response["error"]["code"] == -32800
        assert response["error"]["message"] == "Request cancelled"
        assert "result" not in response
        assert agent.cancelled == ["sess"]
        assert "$/cancel_request" not in caplog.text


@pytest.mark.asyncio
async def test_cancel_request_only_affects_the_targeted_request(server, caplog: pytest.LogCaptureFixture) -> None:
    agent = _BlockingAgent()
    async with AgentSideConnection(cast(Agent, agent), server.server_writer, server.server_reader, listening=True):
        with caplog.at_level(logging.ERROR):
            await _write(server.client_writer, _prompt(1))
            await asyncio.wait_for(agent.started.wait(), timeout=1)
            # Unknown and already-finished ids are ignored without a reply or an error log.
            await _write(server.client_writer, _cancel_request(99))
            await _write(server.client_writer, {"jsonrpc": "2.0", "id": 2, "method": "session/list", "params": {}})
            listed = await _read(server.client_reader)
            await _write(server.client_writer, _cancel_request(2))

            agent.release.set()
            finished = await _read(server.client_reader)

        assert listed == {"jsonrpc": "2.0", "id": 2, "result": {"sessions": []}}
        assert finished == {"jsonrpc": "2.0", "id": 1, "result": {"stopReason": "end_turn"}}
        assert agent.cancelled == []
        assert caplog.text == ""
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(server.client_reader.readline(), timeout=0.1)


@pytest.mark.asyncio
async def test_handler_may_answer_cancel_request_with_a_result(server) -> None:
    started = asyncio.Event()

    class _PartialAgent(TestAgent):
        async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> Any:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return {"stopReason": "cancelled"}

    async with AgentSideConnection(
        cast(Agent, _PartialAgent()), server.server_writer, server.server_reader, listening=True
    ):
        await _write(server.client_writer, _prompt(3))
        await asyncio.wait_for(started.wait(), timeout=1)
        await _write(server.client_writer, _cancel_request(3))

        assert await _read(server.client_reader) == {"jsonrpc": "2.0", "id": 3, "result": {"stopReason": "cancelled"}}


@pytest.mark.asyncio
async def test_internally_cancelled_handler_replies_request_cancelled(server) -> None:
    class _InternallyCancelledAgent(TestAgent):
        async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> Any:
            inner = asyncio.ensure_future(asyncio.Event().wait())
            inner.cancel()
            return await inner

    async with AgentSideConnection(
        cast(Agent, _InternallyCancelledAgent()), server.server_writer, server.server_reader, listening=True
    ):
        await _write(server.client_writer, _prompt(4))

        response = await _read(server.client_reader)
        assert response["id"] == 4
        assert response["error"]["code"] == -32800


@pytest.mark.asyncio
async def test_close_does_not_reply_to_in_flight_requests(server) -> None:
    agent = _BlockingAgent()
    async with AgentSideConnection(
        cast(Agent, agent), server.server_writer, server.server_reader, listening=True
    ) as conn:
        await _write(server.client_writer, _prompt(5))
        await asyncio.wait_for(agent.started.wait(), timeout=1)

        await conn.close()

        assert agent.cancelled == ["sess"]
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(server.client_reader.readline(), timeout=0.1)


@pytest.mark.asyncio
async def test_close_does_not_hang_when_handler_returns_on_cancellation(server) -> None:
    started = asyncio.Event()

    class _PartialAgent(TestAgent):
        async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> Any:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return {"stopReason": "cancelled"}

    async with AgentSideConnection(
        cast(Agent, _PartialAgent()), server.server_writer, server.server_reader, listening=True
    ) as conn:
        await _write(server.client_writer, _prompt(8))
        await asyncio.wait_for(started.wait(), timeout=1)

        closing = asyncio.ensure_future(conn.close())
        done, _ = await asyncio.wait({closing}, timeout=1)
        assert closing in done


@pytest.mark.asyncio
async def test_cancelling_an_outgoing_request_sends_cancel_request(server) -> None:
    async with AgentSideConnection(
        cast(Agent, TestAgent()), server.server_writer, server.server_reader, listening=True
    ) as conn:
        request = asyncio.create_task(
            conn.request_permission(
                session_id="sess",
                tool_call=ToolCallUpdate(tool_call_id="call-1"),
                options=[PermissionOption(option_id="allow", name="Allow", kind="allow_once")],
            )
        )

        outgoing = await _read(server.client_reader)
        assert outgoing["method"] == "session/request_permission"
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        assert await _read(server.client_reader) == {
            "jsonrpc": "2.0",
            "method": "$/cancel_request",
            "params": {"requestId": outgoing["id"]},
        }
        # A late reply to the abandoned request is dropped and the connection stays usable.
        await _write(server.client_writer, {"jsonrpc": "2.0", "id": outgoing["id"], "error": {"code": -32800}})
        await _write(server.client_writer, _prompt(6))
        assert await _read(server.client_reader) == {"jsonrpc": "2.0", "id": 6, "result": {"stopReason": "end_turn"}}


@pytest.mark.asyncio
async def test_completed_outgoing_request_does_not_send_cancel_request(server) -> None:
    async with AgentSideConnection(
        cast(Agent, TestAgent()), server.server_writer, server.server_reader, listening=True
    ) as conn:
        request = asyncio.create_task(
            conn.request_permission(
                session_id="sess",
                tool_call=ToolCallUpdate(tool_call_id="call-1"),
                options=[PermissionOption(option_id="allow", name="Allow", kind="allow_once")],
            )
        )
        outgoing = await _read(server.client_reader)
        await _write(
            server.client_writer,
            {"jsonrpc": "2.0", "id": outgoing["id"], "result": {"outcome": {"outcome": "cancelled"}}},
        )
        response = await asyncio.wait_for(request, timeout=1)

        assert response.outcome.outcome == "cancelled"
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(server.client_reader.readline(), timeout=0.1)


@pytest.mark.asyncio
async def test_cancellation_cascades_between_sdk_peers(server) -> None:
    permission_started = asyncio.Event()
    permission_cancelled = asyncio.Event()

    class _WaitingClient(TestClient):
        async def request_permission(self, session_id: str, tool_call: Any, options: Any, **kwargs: Any) -> Any:
            permission_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                permission_cancelled.set()
                raise

    async with (
        AgentSideConnection(
            cast(Agent, TestAgent()), server.server_writer, server.server_reader, listening=True
        ) as agent_conn,
        ClientSideConnection(_WaitingClient(), server.client_writer, server.client_reader),
    ):
        request = asyncio.create_task(
            agent_conn.request_permission(
                session_id="sess",
                tool_call=ToolCallUpdate(tool_call_id="call-1"),
                options=[PermissionOption(option_id="allow", name="Allow", kind="allow_once")],
            )
        )
        await asyncio.wait_for(permission_started.wait(), timeout=1)

        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        await asyncio.wait_for(permission_cancelled.wait(), timeout=1)
