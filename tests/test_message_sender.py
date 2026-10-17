from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from acp.task import MessageSender, TaskSupervisor


class _ControlledWriter:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error
        self.drain_started = asyncio.Event()
        self.release = asyncio.Event()
        self.writes: list[bytes] = []

    def write(self, payload: bytes) -> None:
        self.writes.append(payload)

    async def drain(self) -> None:
        self.drain_started.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error


async def _wait_for_queue_size(sender: MessageSender, size: int) -> None:
    async def queue_reached_size() -> None:
        while sender._queue.qsize() < size:
            await asyncio.sleep(0)

    await asyncio.wait_for(queue_reached_size(), timeout=1)


async def _assert_tasks_finish(tasks: list[asyncio.Task[Any]]) -> None:
    _done, pending = await asyncio.wait(tasks, timeout=1)
    try:
        assert not pending, f"send tasks still pending: {pending}"
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _close_supervisor(supervisor: TaskSupervisor, tasks: list[asyncio.Task[Any]]) -> None:
    await supervisor.shutdown()
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_writer_failure_settles_active_queued_and_late_sends() -> None:
    error = BrokenPipeError("writer failed")
    writer = _ControlledWriter(error=error)
    supervisor = TaskSupervisor(source="test")
    sender = MessageSender(writer, supervisor)
    tasks: list[asyncio.Task[Any]] = []

    try:
        first = asyncio.create_task(sender.send({"id": 1}), name="send-1")
        tasks.append(first)
        await asyncio.wait_for(writer.drain_started.wait(), timeout=1)

        second = asyncio.create_task(sender.send({"id": 2}), name="send-2")
        tasks.append(second)
        await _wait_for_queue_size(sender, 1)

        writer.release.set()
        with pytest.raises(BrokenPipeError, match="writer failed"):
            await first

        late = asyncio.create_task(sender.send({"id": 3}), name="send-3")
        tasks.append(late)
        await _assert_tasks_finish(tasks)

        for task in (second, late):
            assert isinstance(task.exception(), BrokenPipeError)
            assert str(task.exception()) == "writer failed"
    finally:
        await _close_supervisor(supervisor, tasks)


@pytest.mark.asyncio
async def test_supervisor_shutdown_settles_active_queued_and_late_sends() -> None:
    writer = _ControlledWriter()
    supervisor = TaskSupervisor(source="test")
    sender = MessageSender(writer, supervisor)
    tasks: list[asyncio.Task[Any]] = []

    try:
        first = asyncio.create_task(sender.send({"id": 1}), name="send-1")
        tasks.append(first)
        await asyncio.wait_for(writer.drain_started.wait(), timeout=1)

        second = asyncio.create_task(sender.send({"id": 2}), name="send-2")
        tasks.append(second)
        await _wait_for_queue_size(sender, 1)

        await supervisor.shutdown()
        late = asyncio.create_task(sender.send({"id": 3}), name="send-3")
        tasks.append(late)
        await _assert_tasks_finish(tasks)

        for task in tasks:
            assert isinstance(task.exception(), ConnectionError)
    finally:
        await _close_supervisor(supervisor, tasks)


@pytest.mark.asyncio
async def test_shutdown_before_sender_loop_starts_rejects_later_send() -> None:
    writer = _ControlledWriter()
    supervisor = TaskSupervisor(source="test")
    sender = MessageSender(writer, supervisor)
    tasks: list[asyncio.Task[Any]] = []

    try:
        await supervisor.shutdown()
        late = asyncio.create_task(sender.send({"id": 1}), name="send-after-shutdown")
        tasks.append(late)
        await _assert_tasks_finish(tasks)
        assert isinstance(late.exception(), ConnectionError)
    finally:
        await _close_supervisor(supervisor, tasks)


@pytest.mark.asyncio
async def test_healthy_sends_preserve_order_and_graceful_close() -> None:
    writer = _ControlledWriter()
    supervisor = TaskSupervisor(source="test")
    sender = MessageSender(writer, supervisor)
    tasks: list[asyncio.Task[Any]] = []
    closing: asyncio.Task[None] | None = None

    try:
        tasks.extend([
            asyncio.create_task(sender.send({"id": 1}), name="send-1"),
            asyncio.create_task(sender.send({"id": 2}), name="send-2"),
        ])
        await asyncio.wait_for(writer.drain_started.wait(), timeout=1)
        await _wait_for_queue_size(sender, 1)

        closing = asyncio.create_task(sender.close(), name="sender-close")
        await asyncio.sleep(0)
        assert not closing.done()
        assert [json.loads(payload)["id"] for payload in writer.writes] == [1]

        writer.release.set()
        await asyncio.gather(*tasks)
        await asyncio.wait_for(closing, timeout=1)
        assert [json.loads(payload)["id"] for payload in writer.writes] == [1, 2]

        late = asyncio.create_task(sender.send({"id": 3}), name="send-after-close")
        tasks.append(late)
        await _assert_tasks_finish([late])
        assert isinstance(late.exception(), ConnectionError)
    finally:
        if closing is not None and not closing.done():
            closing.cancel()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await _close_supervisor(supervisor, tasks)
