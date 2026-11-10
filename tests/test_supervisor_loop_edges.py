"""Private edge qualification, separate from reviewed observer candidate."""

from __future__ import annotations

import asyncio
import gc
import weakref

import pytest

from acp.task import TaskSupervisor


def test_no_running_loop_does_not_silently_schedule_on_default_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def work():
        return None

    coroutine = work()
    try:
        with pytest.raises(RuntimeError, match="running event loop"):
            TaskSupervisor(source="edge").create(coroutine)
    finally:
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.run_until_complete(asyncio.sleep(0))
        coroutine.close()
        loop.close()
        asyncio.set_event_loop(None)


@pytest.mark.asyncio
async def test_foreign_loop_future_rejected_before_registration():
    foreign_loop = asyncio.new_event_loop()
    foreign = foreign_loop.create_future()
    supervisor = TaskSupervisor(source="edge")
    try:
        with pytest.raises(ValueError, match="different loop"):
            supervisor.create(foreign)
        assert not supervisor._tasks
    finally:
        foreign.cancel()
        foreign_loop.close()


@pytest.mark.asyncio
async def test_exception_valued_cancelled_error_does_not_escape_callback():
    loop = asyncio.get_running_loop()
    notices = []
    prior = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: notices.append(context))
    supervisor = TaskSupervisor(source="edge")
    future = loop.create_future()
    try:
        supervisor.create(future)
        future.set_exception(asyncio.CancelledError("exception-valued cancellation"))
        await asyncio.sleep(0)
        assert not notices
    finally:
        loop.set_exception_handler(prior)
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_repeated_future_registration_has_independent_task_callbacks():
    supervisor = TaskSupervisor(source="edge")
    future = asyncio.get_running_loop().create_future()
    errors = []
    failure = RuntimeError("registered twice")
    try:
        first = supervisor.create(future, on_error=lambda task, exc: errors.append(("one", task, exc)))
        second = supervisor.create(future, on_error=lambda task, exc: errors.append(("two", task, exc)))
        assert first is not second
        assert isinstance(first, asyncio.Task) and isinstance(second, asyncio.Task)
        assert len(supervisor._tasks) == 2
        future.set_exception(failure)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert errors == [("one", first, failure), ("two", second, failure)]
        assert not supervisor._tasks
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_adopted_task_name_is_not_changed_by_adapter_name():
    supervisor = TaskSupervisor(source="edge")
    task = asyncio.create_task(asyncio.sleep(60), name="caller-owned")
    try:
        adapter = supervisor.create(task, name="supervised")
        assert adapter is not task
        assert adapter.get_name() == "supervised"
        assert task.get_name() == "caller-owned"
    finally:
        await supervisor.shutdown()
    assert task.cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_shutdown_cancels_task_adopted_from_another_supervisor(started):
    first = TaskSupervisor(source="first")
    second = TaskSupervisor(source="second")
    task = asyncio.create_task(asyncio.sleep(60), name="shared")
    try:
        first_adapter = first.create(task)
        second_adapter = second.create(task)
        assert first_adapter is not second_adapter
        if started:
            await asyncio.sleep(0)
        await first.shutdown()
        assert task.cancelled()
        await asyncio.gather(second_adapter, return_exceptions=True)
        await asyncio.sleep(0)
        assert not second._tasks
    finally:
        await first.shutdown()
        await second.shutdown()


@pytest.mark.asyncio
async def test_immediate_shutdown_retrieves_already_failed_future():
    supervisor = TaskSupervisor(source="edge")
    loop = asyncio.get_running_loop()
    notices = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: notices.append(context))
    try:
        future = loop.create_future()
        future.set_exception(RuntimeError("failed before adapter starts"))
        supervisor.create(future)
        await supervisor.shutdown()
        reference = weakref.ref(future)
        del future
        gc.collect()
        await asyncio.sleep(0)
        assert reference() is None
        assert not notices
    finally:
        loop.set_exception_handler(previous)
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_non_awaitable_rejected_before_registration():
    supervisor = TaskSupervisor(source="edge")
    with pytest.raises(TypeError):
        supervisor.create(object())  # type: ignore[arg-type]
    assert not supervisor._tasks
