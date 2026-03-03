"""Tests that MessageSender drains pending futures when the send loop terminates."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from acp.task.sender import MessageSender, _PendingSend
from acp.task.supervisor import TaskSupervisor


class _FailingWriter:
    """A mock StreamWriter that fails after a configurable number of writes."""

    def __init__(self, *, fail_after: int = 0) -> None:
        self._writes = 0
        self._fail_after = fail_after

    def write(self, data: bytes) -> None:
        self._writes += 1
        if self._writes > self._fail_after:
            raise ConnectionResetError("connection lost")

    async def drain(self) -> None:
        pass


@pytest.fixture
def supervisor() -> TaskSupervisor:
    sup = TaskSupervisor(source="test")
    sup.add_error_handler(lambda _task, _exc: None)
    return sup


@pytest.mark.asyncio
async def test_pending_futures_rejected_on_write_error(supervisor: TaskSupervisor) -> None:
    """When the writer raises, queued futures must be rejected (not left hanging)."""
    writer = _FailingWriter(fail_after=0)
    sender = MessageSender(writer, supervisor)  # type: ignore[arg-type]

    # Enqueue several messages directly onto the queue.
    futures: list[asyncio.Future[None]] = []
    loop = asyncio.get_running_loop()
    for i in range(3):
        future: asyncio.Future[None] = loop.create_future()
        await sender._queue.put(
            _PendingSend(
                payload=f'{{"msg":{i}}}\n'.encode(),
                future=future,
            )
        )
        futures.append(future)

    # Give the sender loop time to process and fail.
    await asyncio.sleep(0.15)

    # All futures should be done (either with exception or result), not stuck.
    for future in futures:
        assert future.done(), "Future was left pending after send loop terminated"

    # close() may re-raise the loop's error; suppress it for test cleanup.
    with contextlib.suppress(ConnectionResetError, ConnectionError):
        await sender.close()


@pytest.mark.asyncio
async def test_send_raises_on_write_failure(supervisor: TaskSupervisor) -> None:
    """send() should propagate the error, not hang indefinitely."""
    writer = _FailingWriter(fail_after=0)
    sender = MessageSender(writer, supervisor)  # type: ignore[arg-type]

    with pytest.raises((ConnectionResetError, ConnectionError)):
        await asyncio.wait_for(sender.send({"test": True}), timeout=2.0)

    with contextlib.suppress(ConnectionResetError, ConnectionError):
        await sender.close()


@pytest.mark.asyncio
async def test_successful_send_not_affected(supervisor: TaskSupervisor) -> None:
    """Normal sends should still work when the writer is healthy."""
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()

    sender = MessageSender(writer, supervisor)  # type: ignore[arg-type]

    await asyncio.wait_for(sender.send({"ok": True}), timeout=2.0)
    writer.write.assert_called_once()
    writer.drain.assert_called_once()

    await sender.close()
