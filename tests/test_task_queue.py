"""Tests for InMemoryMessageQueue — publish, consume, close semantics."""

from __future__ import annotations

import asyncio

import pytest

from acp.task import RpcTask, RpcTaskKind
from acp.task.queue import InMemoryMessageQueue


def _make_task(method: str = "test/ping") -> RpcTask:
    return RpcTask(kind=RpcTaskKind.REQUEST, message={"method": method})


@pytest.mark.asyncio
async def test_publish_and_consume() -> None:
    """Published tasks should be yielded by the async iterator."""
    queue = InMemoryMessageQueue()
    task = _make_task("m1")
    await queue.publish(task)
    await queue.close()

    collected: list[RpcTask] = []
    async for item in queue:
        collected.append(item)

    assert len(collected) == 1
    assert collected[0].message["method"] == "m1"


@pytest.mark.asyncio
async def test_fifo_ordering() -> None:
    """Tasks should be consumed in FIFO order."""
    queue = InMemoryMessageQueue()
    for i in range(5):
        await queue.publish(_make_task(f"m{i}"))
    await queue.close()

    methods: list[str] = []
    async for item in queue:
        methods.append(item.message["method"])

    assert methods == [f"m{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_publish_after_close_raises() -> None:
    """Publishing to a closed queue should raise RuntimeError."""
    queue = InMemoryMessageQueue()
    await queue.close()

    with pytest.raises(RuntimeError, match="m[es]*sage queue already closed"):
        await queue.publish(_make_task())


@pytest.mark.asyncio
async def test_close_idempotent() -> None:
    """Closing an already-closed queue should not raise."""
    queue = InMemoryMessageQueue()
    await queue.close()
    await queue.close()


@pytest.mark.asyncio
async def test_task_done_without_get_is_safe() -> None:
    """task_done on an empty queue should not raise (suppresses ValueError)."""
    queue = InMemoryMessageQueue()
    queue.task_done()  # should not raise


@pytest.mark.asyncio
async def test_join_waits_for_task_done() -> None:
    """join() should block until all consumed tasks call task_done()."""
    queue = InMemoryMessageQueue()
    await queue.publish(_make_task())

    joined = False

    async def consumer() -> None:
        nonlocal joined
        async for _ in queue:
            queue.task_done()
        joined = True

    async def closer() -> None:
        await asyncio.sleep(0.05)
        await queue.close()

    await asyncio.gather(consumer(), closer())
    await queue.join()
    assert joined
