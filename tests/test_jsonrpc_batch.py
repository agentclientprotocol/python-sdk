"""Wire tests for JSON-RPC 2.0 batches over the newline-delimited transport.

`Connection` exchanges decoded messages with its transport, so batch handling lives in
`Connection._process_batch`: every request entry is dispatched like a single request, and the
replies are collected into one ordered array. A batch that carries no request stays silent.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from acp._jsonrpc import decode_message
from acp._transport import memory_transport_pair
from acp.connection import Connection
from acp.exceptions import RequestError
from acp.meta import PROTOCOL_METHODS


def _conn(server, handler) -> Connection:
    return Connection(handler, server.server_writer, server.server_reader)


async def _write(server, payload: Any) -> None:
    server.client_writer.write((json.dumps(payload) + "\n").encode())
    await server.client_writer.drain()


async def _read(server) -> Any:
    line = await asyncio.wait_for(server.client_reader.readline(), timeout=1)
    return json.loads(line)


@contextlib.asynccontextmanager
async def _memory(handler) -> AsyncIterator[tuple[Connection, Any]]:
    peer, transport = memory_transport_pair()
    conn = Connection(handler, transport)
    try:
        yield conn, peer
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_batch_requests_return_one_ordered_array(server):
    async def handler(method: str, params: Any, is_notification: bool) -> Any:
        if params == {"slow": True}:
            await asyncio.sleep(0.05)
        return {"echo": params}

    conn = _conn(server, handler)
    try:
        await _write(
            server,
            [
                {"jsonrpc": "2.0", "id": 1, "method": "probe", "params": {"slow": True}},
                {"jsonrpc": "2.0", "id": "two", "method": "probe", "params": {"n": 2}},
            ],
        )

        replies = await _read(server)

        assert isinstance(replies, list), "batch replies must be a single array frame"
        assert [reply["id"] for reply in replies] == [1, "two"]
        assert [reply["result"] for reply in replies] == [{"echo": {"slow": True}}, {"echo": {"n": 2}}]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_notification_only_batch_is_silent(server):
    seen: list[Any] = []

    async def handler(method: str, params: Any, is_notification: bool) -> None:
        if is_notification:
            seen.append(params)

    conn = _conn(server, handler)
    try:
        await _write(
            server,
            [
                {"jsonrpc": "2.0", "method": "note", "params": {"n": 1}},
                {"jsonrpc": "2.0", "method": "note", "params": {"n": 2}},
            ],
        )
        # A follow-up request proves the batch produced no frame at all, rather than a slow one.
        await _write(server, {"jsonrpc": "2.0", "id": 7, "method": "after"})

        assert await _read(server) == {"jsonrpc": "2.0", "id": 7, "result": None}
        assert seen == [{"n": 1}, {"n": 2}]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_mixed_batch_answers_invalid_sibling_in_place(server):
    async def handler(method: str, params: Any, is_notification: bool) -> Any:
        return {"ok": True}

    conn = _conn(server, handler)
    try:
        await _write(server, [1, {"jsonrpc": "2.0", "id": 2, "method": "probe"}])

        replies = await _read(server)

        assert [reply.get("id") for reply in replies] == [None, 2]
        assert replies[0]["error"]["code"] == -32600
        assert replies[1]["result"] == {"ok": True}
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_empty_batch_is_a_single_invalid_request(server):
    conn = _conn(server, lambda method, params, is_notification: None)
    try:
        await _write(server, [])

        reply = await _read(server)

        assert isinstance(reply, dict)
        assert (reply["id"], reply["error"]["code"]) == (None, -32600)
        with pytest.raises(RequestError) as exc_info:
            decode_message("[]")
        assert isinstance(exc_info.value, RequestError)
        assert exc_info.value.code == -32600
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_inbound_response_array_resolves_and_is_not_echoed():
    async def handler(method: str, params: Any, is_notification: bool) -> Any:
        return None

    async with _memory(handler) as (conn, peer):
        pending = asyncio.create_task(conn.send_request("probe"))
        request_id = (await asyncio.wait_for(peer.receive(), timeout=1))["id"]

        await peer.send([
            {"jsonrpc": "2.0", "id": request_id, "result": {"answered": True}},
            {"jsonrpc": "2.0", "id": 9, "method": "probe"},
        ])

        assert await asyncio.wait_for(pending, timeout=1) == {"answered": True}
        assert await asyncio.wait_for(peer.receive(), timeout=1) == [{"jsonrpc": "2.0", "id": 9, "result": None}]


@pytest.mark.asyncio
async def test_batch_handler_can_reenter_the_connection():
    async def handler(method: str, params: Any, is_notification: bool) -> Any:
        # Runs inside ``_execute_request``'s own task, so the receive loop stays free to route the
        # peer's answer back into this coroutine while the batch itself is still unanswered.
        answer = await conn.send_request("peer/question", {"n": params["n"]})
        return {"n": params["n"], "peer": answer}

    async with _memory(handler) as (conn, peer):
        await peer.send([
            {"jsonrpc": "2.0", "id": 1, "method": "probe", "params": {"n": 1}},
            {"jsonrpc": "2.0", "id": 2, "method": "probe", "params": {"n": 2}},
        ])

        for _ in range(2):
            nested = await asyncio.wait_for(peer.receive(), timeout=1)
            assert nested["method"] == "peer/question"
            await peer.send({"jsonrpc": "2.0", "id": nested["id"], "result": {"answered": nested["params"]["n"]}})

        replies = await asyncio.wait_for(peer.receive(), timeout=1)

        assert isinstance(replies, list), "both entries must travel back in one array frame"
        assert [reply["result"] for reply in replies] == [
            {"n": 1, "peer": {"answered": 1}},
            {"n": 2, "peer": {"answered": 2}},
        ]


@pytest.mark.asyncio
async def test_cancelled_batch_entry_answers_once_and_keeps_its_sibling_result():
    started = asyncio.Event()

    async def handler(method: str, params: Any, is_notification: bool) -> Any:
        if params == {"slow": True}:
            started.set()
            await asyncio.sleep(5)
        return {"echo": params}

    async with _memory(handler) as (_, peer):
        await peer.send([
            {"jsonrpc": "2.0", "id": "slow", "method": "probe", "params": {"slow": True}},
            {"jsonrpc": "2.0", "id": "fast", "method": "probe", "params": {"slow": False}},
        ])
        await asyncio.wait_for(started.wait(), timeout=1)
        # The cancel names an id already registered from inside the same batch.
        await peer.send([
            {"jsonrpc": "2.0", "method": PROTOCOL_METHODS["cancel_request"], "params": {"requestId": "slow"}}
        ])

        replies = await asyncio.wait_for(peer.receive(), timeout=1)

        assert [reply["id"] for reply in replies] == ["slow", "fast"]
        assert (replies[0]["error"]["code"], replies[0]["error"]["message"]) == (-32800, "Request cancelled")
        assert "result" not in replies[0]
        assert replies[1] == {"jsonrpc": "2.0", "id": "fast", "result": {"echo": {"slow": False}}}
        # A single array frame, so the cancelled id is answered exactly once.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(peer.receive(), timeout=0.1)
