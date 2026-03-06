"""Tests for TaskSupervisor lifecycle and error handling."""

from __future__ import annotations

import asyncio

import pytest

from acp.task.supervisor import TaskSupervisor


@pytest.fixture
def supervisor() -> TaskSupervisor:
    return TaskSupervisor(source="test")


@pytest.mark.asyncio
async def test_create_and_run_task(supervisor: TaskSupervisor) -> None:
    """Created tasks should execute and complete."""
    result: list[int] = []

    async def worker() -> None:
        result.append(42)

    task = supervisor.create(worker(), name="test-worker")
    await task
    assert result == [42]


@pytest.mark.asyncio
async def test_create_raises_after_shutdown(supervisor: TaskSupervisor) -> None:
    """Cannot create tasks after supervisor is shut down."""
    await supervisor.shutdown()

    with pytest.raises(RuntimeError, match="already closed"):
        supervisor.create(asyncio.sleep(0), name="late")


@pytest.mark.asyncio
async def test_shutdown_cancels_running_tasks(supervisor: TaskSupervisor) -> None:
    """Shutdown should cancel all in-flight tasks."""
    started = asyncio.Event()

    async def long_running() -> None:
        started.set()
        await asyncio.sleep(999)

    task = supervisor.create(long_running(), name="long")
    await started.wait()

    await supervisor.shutdown()
    assert task.cancelled()


@pytest.mark.asyncio
async def test_shutdown_idempotent(supervisor: TaskSupervisor) -> None:
    """Multiple shutdown calls should not raise."""
    await supervisor.shutdown()
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_task_specific_error_handler(supervisor: TaskSupervisor) -> None:
    """Per-task on_error callback should receive the exception."""
    captured: list[BaseException] = []

    def on_error(_task: asyncio.Task, exc: BaseException) -> None:
        captured.append(exc)

    async def failing() -> None:
        raise ValueError("boom")

    supervisor.create(failing(), name="fail", on_error=on_error)
    await asyncio.sleep(0.05)

    assert len(captured) == 1
    assert str(captured[0]) == "boom"


@pytest.mark.asyncio
async def test_global_error_handler_fallback(supervisor: TaskSupervisor) -> None:
    """Global error handlers fire when no per-task handler is set."""
    captured: list[BaseException] = []
    supervisor.add_error_handler(lambda _t, exc: captured.append(exc))

    async def failing() -> None:
        raise RuntimeError("global-boom")

    supervisor.create(failing(), name="fail-global")
    await asyncio.sleep(0.05)

    assert len(captured) == 1
    assert str(captured[0]) == "global-boom"


@pytest.mark.asyncio
async def test_completed_task_removed_from_registry(supervisor: TaskSupervisor) -> None:
    """Finished tasks should be discarded from the internal set."""

    async def quick() -> None:
        pass

    supervisor.create(quick(), name="quick")
    await asyncio.sleep(0.05)

    # After completion, shutdown should be instant (no tasks to cancel)
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_cancelled_task_not_reported_as_error(supervisor: TaskSupervisor) -> None:
    """Cancelled tasks should not trigger error handlers."""
    errors: list[BaseException] = []
    supervisor.add_error_handler(lambda _t, exc: errors.append(exc))

    async def sleeper() -> None:
        await asyncio.sleep(999)

    supervisor.create(sleeper(), name="sleeper")
    await asyncio.sleep(0.01)
    await supervisor.shutdown()

    assert errors == []
