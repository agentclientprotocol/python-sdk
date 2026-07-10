"""Unhandled RPC handler exceptions must be logged instead of silently swallowed.

Requests already returned a JSON-RPC -32603 error but discarded the original
traceback; notifications suppressed the exception entirely. Both now log the
underlying exception so integrators (e.g. Sentry via its logging integration)
can see server-side handler crashes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from acp.connection import Connection, MethodHandler
from acp.exceptions import RequestError


class _RecordingSender:
    """Duck-typed MessageSender that records outgoing frames instead of writing them."""

    def __init__(self, writer: asyncio.StreamWriter, supervisor: Any) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        pass


def _make_connection(handler: MethodHandler) -> tuple[Connection, _RecordingSender]:
    captured: dict[str, _RecordingSender] = {}

    def sender_factory(writer: asyncio.StreamWriter, supervisor: Any) -> _RecordingSender:
        captured["sender"] = _RecordingSender(writer, supervisor)
        return captured["sender"]

    conn = Connection(handler, MagicMock(), MagicMock(), sender_factory=sender_factory, listening=False)
    return conn, captured["sender"]


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
async def test_run_request_unhandled_exception_is_logged_and_returned_as_internal_error(caplog):
    conn, sender = _make_connection(_raising_handler)
    request = {"jsonrpc": "2.0", "id": 7, "method": "explode", "params": None}

    try:
        with caplog.at_level(logging.ERROR), pytest.raises(RequestError) as exc_info:
            await conn._run_request(request)
    finally:
        await conn.close()

    # The handler exception is re-raised as a JSON-RPC internal error...
    raised = exc_info.value
    assert isinstance(raised, RequestError)
    assert raised.code == -32603
    assert raised.data == {"details": "kaboom"}

    # ...and exactly one error frame carrying the handler's message is written to the peer.
    assert len(sender.sent) == 1
    response = sender.sent[0]
    assert response["id"] == 7
    assert "result" not in response
    assert response["error"] == {"code": -32603, "message": "Internal error", "data": {"details": "kaboom"}}

    # The original exception is logged, not discarded by `raise err from None`.
    _assert_logged_runtime_error(caplog, "explode")


@pytest.mark.asyncio
async def test_run_notification_unhandled_exception_is_logged_and_not_answered(caplog):
    conn, sender = _make_connection(_raising_handler)
    notification = {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": "s1"}}

    try:
        with caplog.at_level(logging.ERROR):
            result = await conn._run_notification(notification)
    finally:
        await conn.close()

    # A notification has no response: the error is neither raised nor written to the wire.
    assert result is None
    assert sender.sent == []

    # It must still be logged — previously contextlib.suppress dropped it silently.
    _assert_logged_runtime_error(caplog, "session/cancel")
