from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from acp.connection import Connection
from acp.exceptions import RequestError


async def _noop_handler(method: str, params: Any, is_notification: bool) -> Any:
    return None


def _make_connection(
    *,
    limit: int = 128,
    receive_timeout: float | None = None,
) -> tuple[Connection, asyncio.StreamReader]:
    reader = asyncio.StreamReader(limit=limit)
    transport = MagicMock()
    transport.is_closing.return_value = False
    protocol = AsyncMock()
    writer = asyncio.StreamWriter(transport, protocol, reader, asyncio.get_running_loop())
    conn = Connection(_noop_handler, writer, reader, listening=False, receive_timeout=receive_timeout)
    return conn, reader


@pytest.mark.asyncio
async def test_receive_loop_handles_oversized_frame(caplog: pytest.LogCaptureFixture) -> None:
    conn, reader = _make_connection(limit=128)
    processed: list[str] = []

    def tracking_process(message: dict[str, Any]) -> None:
        processed.append(message["method"])

    conn._process_message = tracking_process  # type: ignore[method-assign]
    oversized = {"jsonrpc": "2.0", "method": "too-large", "params": {"data": "X" * 256}}
    survivor = {"jsonrpc": "2.0", "method": "survivor"}
    reader.feed_data(json.dumps(oversized).encode() + b"\n" + json.dumps(survivor).encode() + b"\n")
    reader.feed_eof()

    with caplog.at_level("WARNING"):
        await conn._receive_loop()
    await conn.close()

    assert processed == ["too-large", "survivor"]
    assert "oversized JSON-RPC frame" not in caplog.text


@pytest.mark.asyncio
async def test_receive_loop_handles_consecutive_oversized_frames() -> None:
    conn, reader = _make_connection(limit=128)
    processed: list[str] = []

    def tracking_process(message: dict[str, Any]) -> None:
        processed.append(message["method"])

    conn._process_message = tracking_process  # type: ignore[method-assign]
    for index in range(2):
        oversized = {"jsonrpc": "2.0", "method": f"too-large-{index}", "params": {"data": "Y" * 256}}
        reader.feed_data(json.dumps(oversized).encode() + b"\n")
    survivor = {"jsonrpc": "2.0", "method": "survivor"}
    reader.feed_data(json.dumps(survivor).encode() + b"\n")
    reader.feed_eof()

    await conn._receive_loop()
    await conn.close()

    assert processed == ["too-large-0", "too-large-1", "survivor"]


@pytest.mark.asyncio
async def test_receive_loop_handles_eof_during_oversized_frame() -> None:
    conn, reader = _make_connection(limit=64)
    reader.feed_data(b"X" * 256)
    reader.feed_eof()

    await conn._receive_loop()
    await conn.close()

    assert conn._disconnected is True


@pytest.mark.asyncio
async def test_receive_loop_keeps_timeout_semantics() -> None:
    conn, _reader = _make_connection(receive_timeout=0.01)

    with pytest.raises(RequestError) as exc_info:
        await conn._receive_loop()
    await conn.close()

    exc = exc_info.value
    assert isinstance(exc, RequestError)
    assert str(exc) == "Internal error"
    assert exc.data == {"details": "Agent timeout"}


@pytest.mark.asyncio
async def test_receive_loop_keeps_timeout_semantics_while_reading_oversized_frame() -> None:
    conn, reader = _make_connection(limit=64, receive_timeout=0.01)
    reader.feed_data(b"X" * 256)

    with pytest.raises(RequestError) as exc_info:
        await conn._receive_loop()
    await conn.close()

    exc = exc_info.value
    assert isinstance(exc, RequestError)
    assert exc.data == {"details": "Agent timeout"}


@pytest.mark.asyncio
async def test_receive_loop_does_not_swallow_unrelated_reader_error() -> None:
    conn, reader = _make_connection()
    reader.set_exception(ValueError("reader failed"))

    with pytest.raises(ValueError, match="reader failed"):
        await conn._receive_loop()
    await conn.close()


@pytest.mark.asyncio
async def test_receive_loop_ignores_frames_that_are_not_json_objects() -> None:
    """A line can be valid JSON and still not be a JSON-RPC message.

    Unparsable input is already skipped by ``NdjsonTransport.receive``. These frames parse
    fine but are not objects: an array, a number, a string and ``null``. Without the object
    guard they reach ``Connection._process_message`` -- which calls ``message.get(...)`` --
    or, for ``null``, are read as the transport's EOF signal. Either way the connection dies
    and the valid frame queued behind them is never handled.
    """
    conn, reader = _make_connection()
    processed: list[str] = []

    def tracking_process(message: dict[str, Any]) -> None:
        processed.append(message["method"])

    conn._process_message = tracking_process  # type: ignore[method-assign]
    non_objects = b"\n".join([b"[]", b"123", b'"x"', b"null", b"", b"not json at all"])
    survivor = {"jsonrpc": "2.0", "method": "survivor"}
    reader.feed_data(non_objects + b"\n" + json.dumps(survivor).encode() + b"\n")
    reader.feed_eof()

    await conn._receive_loop()
    await conn.close()

    assert processed == ["survivor"]


@pytest.mark.asyncio
async def test_receive_loop_ignores_responses_with_an_invalid_id() -> None:
    """A response whose ``id`` is not a ``RequestId`` must not answer anyone.

    ``_handle_response`` looks the id up in ``_pending``. An array or an object raises there,
    which escapes the receive loop, disconnects, and leaves the valid frame behind it
    unhandled. ``true`` and ``1.0`` are worse than a crash: they are equal to the integer
    ``1``, so they resolve a different request's future -- the aliasing ``_track_incoming``
    already refuses on the way out.
    """
    conn, reader = _make_connection()
    original = asyncio.get_running_loop().create_future()
    conn._pending[1] = original
    # The real dispatch is left in place: a response reaches ``_handle_response``, which is
    # where the id is used. The loop is not fed EOF, so it stays alive and ``_disconnect``
    # does not reject the pending future before it can be inspected.
    handled: list[str] = []
    real_track = conn._track_incoming

    def tracking_track(request_id: Any, task: asyncio.Task[Any]) -> None:
        handled.append("request")
        real_track(request_id, task)

    conn._track_incoming = tracking_track  # type: ignore[method-assign]

    invalid = b"\n".join([
        json.dumps({"jsonrpc": "2.0", "id": [], "result": "array"}).encode(),
        json.dumps({"jsonrpc": "2.0", "id": {}, "result": "object"}).encode(),
        json.dumps({"jsonrpc": "2.0", "id": True, "result": "aliased"}).encode(),
        json.dumps({"jsonrpc": "2.0", "id": 1.0, "result": "aliased"}).encode(),
    ])
    survivor = {"jsonrpc": "2.0", "method": "survivor", "id": 2}
    reader.feed_data(invalid + b"\n" + json.dumps(survivor).encode() + b"\n")

    loop_task = asyncio.create_task(conn._receive_loop())
    for _ in range(200):
        await asyncio.sleep(0)
        if handled and 1 not in conn._incoming:
            break

    # None of the four bad ids answered the integer-1 request, and the loop kept going: the
    # valid request behind them was dispatched. Request 2 answers nothing and is cleaned up,
    # so only the unanswered request 1 remains pending.
    assert not original.done()
    assert sorted(conn._pending) == [1]
    assert handled == ["request"]

    loop_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await loop_task

    conn._pending.clear()
    await conn.close()
