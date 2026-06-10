from __future__ import annotations

import asyncio
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
async def test_receive_loop_recovers_from_oversized_frame(caplog: pytest.LogCaptureFixture) -> None:
    conn, reader = _make_connection(limit=128)
    processed: list[str] = []

    async def tracking_process(message: dict[str, Any]) -> None:
        processed.append(message["method"])

    conn._process_message = tracking_process  # type: ignore[method-assign]
    oversized = {"jsonrpc": "2.0", "method": "too-large", "params": {"data": "X" * 256}}
    survivor = {"jsonrpc": "2.0", "method": "survivor"}
    reader.feed_data(json.dumps(oversized).encode() + b"\n" + json.dumps(survivor).encode() + b"\n")
    reader.feed_eof()

    with caplog.at_level("WARNING"):
        await conn._receive_loop()
    await conn.close()

    assert processed == ["survivor"]
    assert any("oversized JSON-RPC frame" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_receive_loop_recovers_from_consecutive_oversized_frames() -> None:
    conn, reader = _make_connection(limit=128)
    processed: list[str] = []

    async def tracking_process(message: dict[str, Any]) -> None:
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

    assert processed == ["survivor"]


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
async def test_receive_loop_does_not_swallow_unrelated_reader_error() -> None:
    conn, reader = _make_connection()
    reader.set_exception(ValueError("reader failed"))

    with pytest.raises(ValueError, match="reader failed"):
        await conn._receive_loop()
    await conn.close()
