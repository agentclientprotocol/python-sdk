"""Unhandled RPC handler exceptions must be logged instead of silently swallowed.

Requests already returned a JSON-RPC -32603 error but discarded the original
traceback; notifications suppressed the exception entirely. Both now log the
underlying exception so integrators (e.g. Sentry via its logging integration)
can see server-side handler crashes.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from acp._transport import Transport
from acp.connection import Connection, MethodHandler


class _RecordingTransport:
    """Message transport that records outgoing frames."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def receive(self) -> dict[str, Any] | None:
        return None

    async def close(self) -> None:
        pass


def _make_connection(handler: MethodHandler) -> tuple[Connection, _RecordingTransport]:
    transport = _RecordingTransport()
    conn = Connection(handler, transport, listening=False)
    return conn, transport


async def _raising_handler(method: str, params: Any, is_notification: bool) -> Any:
    raise RuntimeError("kaboom")


def _assert_logged_runtime_error(caplog: pytest.LogCaptureFixture, method: str) -> None:
    records = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR and record.exc_info and f"method={method}" in record.getMessage()
    ]
    assert len(records) == 1, f"expected exactly one logged error for method={method}"
    exc_info = records[0].exc_info
    assert exc_info is not None
    logged = exc_info[1]
    assert isinstance(logged, RuntimeError)
    assert str(logged) == "kaboom"


@pytest.mark.asyncio
async def test_run_request_unhandled_exception_is_logged_and_sent_as_internal_error(caplog):
    conn, transport = _make_connection(_raising_handler)
    request = {"jsonrpc": "2.0", "id": 7, "method": "explode", "params": None}

    try:
        with caplog.at_level(logging.ERROR):
            await conn._run_request(request)
    finally:
        await conn.close()

    # A handled application exception becomes exactly one JSON-RPC error frame.
    assert len(transport.sent) == 1
    response = transport.sent[0]
    assert response["id"] == 7
    assert "result" not in response
    assert response["error"] == {"code": -32603, "message": "Internal error", "data": {"details": "kaboom"}}

    # The original exception is logged, not discarded by `raise err from None`.
    _assert_logged_runtime_error(caplog, "explode")


@pytest.mark.asyncio
async def test_run_notification_unhandled_exception_is_logged_and_not_answered(caplog):
    conn, transport = _make_connection(_raising_handler)
    notification = {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": "s1"}}

    try:
        with caplog.at_level(logging.ERROR):
            result = await conn._run_notification(notification)
    finally:
        await conn.close()

    # A notification has no response: the error is neither raised nor written to the wire.
    assert result is None
    assert transport.sent == []

    # It must still be logged — previously contextlib.suppress dropped it silently.
    _assert_logged_runtime_error(caplog, "session/cancel")


@pytest.mark.asyncio
async def test_response_send_failure_is_not_mapped_to_a_handler_error(caplog):
    class _FailingTransport(_RecordingTransport):
        async def send(self, message: dict[str, Any]) -> None:
            raise ConnectionError("send failed")

    async def successful_handler(method: str, params: Any, is_notification: bool) -> Any:
        return {"ok": True}

    transport: Transport = _FailingTransport()
    conn = Connection(successful_handler, transport, listening=False)
    request = {"jsonrpc": "2.0", "id": 8, "method": "succeed", "params": None}

    try:
        with caplog.at_level(logging.ERROR), pytest.raises(ConnectionError, match="send failed"):
            await conn._run_request(request)
    finally:
        await conn.close()

    assert "Unhandled error while handling request method=succeed" not in caplog.text
