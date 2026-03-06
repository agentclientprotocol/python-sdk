"""Tests for DefaultMessageDispatcher — request/notification routing."""

from __future__ import annotations

import asyncio

import pytest

from acp.task import RpcTask, RpcTaskKind
from acp.task.dispatcher import DefaultMessageDispatcher
from acp.task.queue import InMemoryMessageQueue
from acp.task.state import InMemoryMessageStateStore
from acp.task.supervisor import TaskSupervisor


@pytest.fixture
def supervisor() -> TaskSupervisor:
    sup = TaskSupervisor(source="test")
    sup.add_error_handler(lambda _t, _e: None)
    return sup


@pytest.fixture
def store() -> InMemoryMessageStateStore:
    return InMemoryMessageStateStore()


@pytest.fixture
def queue() -> InMemoryMessageQueue:
    return InMemoryMessageQueue()


@pytest.mark.asyncio
async def test_dispatch_request(
    supervisor: TaskSupervisor,
    store: InMemoryMessageStateStore,
    queue: InMemoryMessageQueue,
) -> None:
    """Dispatcher should route REQUEST tasks to the request runner."""
    results: list[dict] = []

    async def request_runner(msg: dict) -> dict:
        results.append(msg)
        return {"ok": True}

    async def notification_runner(msg: dict) -> None:
        pass

    dispatcher = DefaultMessageDispatcher(
        queue=queue,
        supervisor=supervisor,
        store=store,
        request_runner=request_runner,
        notification_runner=notification_runner,
    )
    dispatcher.start()

    await queue.publish(RpcTask(kind=RpcTaskKind.REQUEST, message={"method": "test/req", "params": {}}))
    await asyncio.sleep(0.1)
    await queue.close()
    await dispatcher.stop()
    await supervisor.shutdown()

    assert len(results) == 1
    assert results[0]["method"] == "test/req"


@pytest.mark.asyncio
async def test_dispatch_notification(
    supervisor: TaskSupervisor,
    store: InMemoryMessageStateStore,
    queue: InMemoryMessageQueue,
) -> None:
    """Dispatcher should route NOTIFICATION tasks to the notification runner."""
    notifications: list[dict] = []

    async def request_runner(msg: dict) -> dict:
        return {}

    async def notification_runner(msg: dict) -> None:
        notifications.append(msg)

    dispatcher = DefaultMessageDispatcher(
        queue=queue,
        supervisor=supervisor,
        store=store,
        request_runner=request_runner,
        notification_runner=notification_runner,
    )
    dispatcher.start()

    await queue.publish(RpcTask(kind=RpcTaskKind.NOTIFICATION, message={"method": "session/update"}))
    await asyncio.sleep(0.1)
    await queue.close()
    await dispatcher.stop()
    await supervisor.shutdown()

    assert len(notifications) == 1
    assert notifications[0]["method"] == "session/update"


@pytest.mark.asyncio
async def test_start_twice_raises(
    supervisor: TaskSupervisor,
    store: InMemoryMessageStateStore,
    queue: InMemoryMessageQueue,
) -> None:
    """Starting the dispatcher twice should raise."""

    async def noop(msg: dict) -> None:
        pass

    dispatcher = DefaultMessageDispatcher(
        queue=queue,
        supervisor=supervisor,
        store=store,
        request_runner=noop,
        notification_runner=noop,
    )
    dispatcher.start()

    with pytest.raises(RuntimeError, match="already started"):
        dispatcher.start()

    await queue.close()
    await dispatcher.stop()
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_failed_request_updates_store(
    supervisor: TaskSupervisor,
    store: InMemoryMessageStateStore,
    queue: InMemoryMessageQueue,
) -> None:
    """When the request runner raises, the store should record the failure."""

    async def failing_runner(msg: dict) -> dict:
        raise ValueError("handler error")

    async def notification_runner(msg: dict) -> None:
        pass

    dispatcher = DefaultMessageDispatcher(
        queue=queue,
        supervisor=supervisor,
        store=store,
        request_runner=failing_runner,
        notification_runner=notification_runner,
    )
    dispatcher.start()

    await queue.publish(RpcTask(kind=RpcTaskKind.REQUEST, message={"method": "test/fail", "params": None}))
    await asyncio.sleep(0.15)
    await queue.close()
    await dispatcher.stop()
    await supervisor.shutdown()

    # The store should have one incoming record with failed status
    assert len(store._incoming) == 1
    assert store._incoming[0].status == "failed"
    assert isinstance(store._incoming[0].error, ValueError)


@pytest.mark.asyncio
async def test_multiple_tasks_dispatched(
    supervisor: TaskSupervisor,
    store: InMemoryMessageStateStore,
    queue: InMemoryMessageQueue,
) -> None:
    """Multiple tasks should all be dispatched and processed."""
    processed: list[str] = []

    async def request_runner(msg: dict) -> dict:
        processed.append(msg["method"])
        return {}

    async def notification_runner(msg: dict) -> None:
        processed.append(msg["method"])

    dispatcher = DefaultMessageDispatcher(
        queue=queue,
        supervisor=supervisor,
        store=store,
        request_runner=request_runner,
        notification_runner=notification_runner,
    )
    dispatcher.start()

    await queue.publish(RpcTask(kind=RpcTaskKind.REQUEST, message={"method": "r1"}))
    await queue.publish(RpcTask(kind=RpcTaskKind.NOTIFICATION, message={"method": "n1"}))
    await queue.publish(RpcTask(kind=RpcTaskKind.REQUEST, message={"method": "r2"}))

    await asyncio.sleep(0.15)
    await queue.close()
    await dispatcher.stop()
    await supervisor.shutdown()

    assert sorted(processed) == ["n1", "r1", "r2"]
