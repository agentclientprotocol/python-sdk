"""Tests for connection recovery from oversized messages (Issue #62).

Verifies that the connection's _receive_loop gracefully handles messages
that exceed the StreamReader buffer limit by catching LimitOverrunError
and continuing to process subsequent valid messages.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from acp.connection import Connection


async def _noop_handler(method: str, params: Any, is_notification: bool) -> Any:
    return None


def _make_connection(
    limit: int = 4096,
    handler: Any = None,
) -> tuple[Connection, asyncio.StreamReader]:
    """Create a Connection with a StreamReader of the given buffer limit."""
    reader = asyncio.StreamReader(limit=limit)
    transport = MagicMock()
    transport.is_closing.return_value = False
    protocol = AsyncMock()
    writer = asyncio.StreamWriter(transport, protocol, reader, asyncio.get_running_loop())
    conn = Connection(handler or _noop_handler, writer, reader, listening=False)
    return conn, reader


@pytest.mark.asyncio
async def test_receive_loop_recovers_from_oversized_message() -> None:
    """The receive loop should skip oversized messages and continue processing."""
    conn, reader = _make_connection(limit=128)

    # Track which messages were processed
    processed: list[dict] = []
    original_process = conn._process_message

    async def tracking_process(message: dict[str, Any]) -> None:
        processed.append(message)
        await original_process(message)

    conn._process_message = tracking_process  # type: ignore[assignment]

    # Feed an oversized message (exceeds 128-byte limit)
    oversized_msg = json.dumps({"jsonrpc": "2.0", "method": "test.oversized", "params": {"data": "X" * 200}})
    reader.feed_data((oversized_msg + "\n").encode())

    # Feed a normal message after the oversized one
    normal_msg = json.dumps({"jsonrpc": "2.0", "method": "test.normal", "params": {}})
    reader.feed_data((normal_msg + "\n").encode())

    # Signal EOF
    reader.feed_eof()

    # Run the receive loop
    await conn._receive_loop()

    # The oversized message should have been skipped, but the normal one processed
    assert len(processed) == 1
    assert processed[0]["method"] == "test.normal"
    await conn.close()


@pytest.mark.asyncio
async def test_receive_loop_does_not_crash_on_limit_overrun() -> None:
    """The receive loop must not raise LimitOverrunError or ValueError."""
    conn, reader = _make_connection(limit=64)

    # Feed only an oversized message and then EOF
    oversized = "X" * 200 + "\n"
    reader.feed_data(oversized.encode())
    reader.feed_eof()

    # This should NOT raise — the error should be caught and logged
    await conn._receive_loop()
    await conn.close()


@pytest.mark.asyncio
async def test_receive_loop_handles_normal_messages() -> None:
    """Sanity check: normal messages within the limit are processed correctly."""
    processed: list[dict] = []
    conn, reader = _make_connection(limit=4096)

    original_process = conn._process_message

    async def tracking_process(message: dict[str, Any]) -> None:
        processed.append(message)
        await original_process(message)

    conn._process_message = tracking_process  # type: ignore[assignment]

    msg1 = json.dumps({"jsonrpc": "2.0", "method": "test.one", "params": {}})
    msg2 = json.dumps({"jsonrpc": "2.0", "method": "test.two", "params": {}})
    reader.feed_data((msg1 + "\n" + msg2 + "\n").encode())
    reader.feed_eof()

    await conn._receive_loop()

    assert len(processed) == 2
    assert processed[0]["method"] == "test.one"
    assert processed[1]["method"] == "test.two"
    await conn.close()


@pytest.mark.asyncio
async def test_receive_loop_recovers_from_multiple_oversized_messages() -> None:
    """Multiple consecutive oversized messages should all be skipped gracefully."""
    conn, reader = _make_connection(limit=128)

    processed: list[dict] = []
    original_process = conn._process_message

    async def tracking_process(message: dict[str, Any]) -> None:
        processed.append(message)
        await original_process(message)

    conn._process_message = tracking_process  # type: ignore[assignment]

    # Feed two oversized messages
    for i in range(2):
        oversized = json.dumps({"jsonrpc": "2.0", "method": f"test.big{i}", "params": {"data": "Y" * 200}})
        reader.feed_data((oversized + "\n").encode())

    # Feed a normal message after both oversized ones
    normal = json.dumps({"jsonrpc": "2.0", "method": "test.survivor", "params": {}})
    reader.feed_data((normal + "\n").encode())
    reader.feed_eof()

    await conn._receive_loop()

    # Only the normal message should be processed
    assert len(processed) == 1
    assert processed[0]["method"] == "test.survivor"
    await conn.close()


@pytest.mark.asyncio
async def test_receive_loop_logs_warning_on_oversized_message(caplog: pytest.LogCaptureFixture) -> None:
    """A warning should be logged when an oversized message is skipped."""
    conn, reader = _make_connection(limit=64)

    oversized = json.dumps({"jsonrpc": "2.0", "method": "test.huge", "params": {"data": "Z" * 200}})
    reader.feed_data((oversized + "\n").encode())
    reader.feed_eof()

    with caplog.at_level("WARNING"):
        await conn._receive_loop()

    assert any(
        "oversized" in record.message.lower() or "buffer limit" in record.message.lower() for record in caplog.records
    )
    await conn.close()


@pytest.mark.asyncio
async def test_unrelated_value_error_is_not_swallowed() -> None:
    """A ValueError that is NOT from a limit overrun should propagate."""
    conn, reader = _make_connection(limit=4096)

    # Inject an unrelated ValueError into the reader
    reader.set_exception(ValueError("completely unrelated error"))

    with pytest.raises(ValueError, match="completely unrelated"):
        await conn._receive_loop()

    await conn.close()
