"""Observer Awaitable results must not break JSON-RPC delivery."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from acp._transport import memory_transport_pair
from acp.connection import Connection


class CustomAwaitable:
    def __init__(self, future: asyncio.Future[None]) -> None:
        self.future = future

    def __await__(self):
        return self.future.__await__()


def observer_result(kind: str, future: asyncio.Future[None]):
    if kind == "done_future":
        future.set_result(None)
    if kind == "custom":
        return CustomAwaitable(future), None
    if kind == "task":

        async def observe() -> None:
            await future

        task = asyncio.create_task(observe())
        return task, task
    return future, None


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
@pytest.mark.parametrize("kind", ["done_future", "pending_future", "task", "custom"])
async def test_awaitable_observer_preserves_delivery(direction: str, kind: str) -> None:
    source, peer = memory_transport_pair()
    handled = asyncio.Event()
    observed = []
    future = asyncio.get_running_loop().create_future()
    result, task = observer_result(kind, future)

    def observer(event):
        observed.append(event.direction.value)
        return result

    async def handler(method, params, notification):
        handled.set()

    connection = Connection(handler, source, observers=[observer], listening=direction == "incoming")
    error = None
    try:
        if direction == "outgoing":
            try:
                await connection.send_notification("example", {"value": 1})
            except TypeError as exc:
                error = exc
            message = await asyncio.wait_for(peer.receive(), 1)
            assert message["method"] == "example"
            assert error is None, "a supported observer Awaitable must not fail an already-sent notification"
        else:
            await peer.send({"jsonrpc": "2.0", "method": "example", "params": {"value": 1}})
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(handled.wait(), 0.1)
            assert handled.is_set(), "a supported observer Awaitable must not disconnect before dispatch"
        assert observed == [direction]
        if not future.done():
            future.set_result(None)
        if task is not None:
            await task
        await asyncio.sleep(0)
    finally:
        await connection.close()
        await peer.close()
        if not future.done():
            future.cancel()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
@pytest.mark.parametrize("kind", ["sync", "coroutine", "sync_error"])
async def test_existing_observer_controls(direction: str, kind: str, caplog) -> None:
    source, peer = memory_transport_pair()
    handled, observed = asyncio.Event(), asyncio.Event()

    async def async_observer(event):
        observed.set()

    def sync_observer(event):
        observed.set()
        if kind == "sync_error":
            raise RuntimeError("synthetic observer failure")

    async def handler(method, params, notification):
        handled.set()

    observer = async_observer if kind == "coroutine" else sync_observer
    connection = Connection(handler, source, observers=[observer], listening=direction == "incoming")
    try:
        if direction == "outgoing":
            await connection.send_notification("example")
            assert (await asyncio.wait_for(peer.receive(), 1))["method"] == "example"
        else:
            await peer.send({"jsonrpc": "2.0", "method": "example"})
            await asyncio.wait_for(handled.wait(), 1)
        await asyncio.wait_for(observed.wait(), 1)
        if kind == "sync_error":
            assert "Stream observer failed" in caplog.text
    finally:
        await connection.close()
        await peer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["future", "custom", "task"])
@pytest.mark.parametrize("started", [False, True])
async def test_shutdown_cancels_pending_observer_awaitable(kind: str, started: bool) -> None:
    source, peer = memory_transport_pair()
    future = asyncio.get_running_loop().create_future()
    result, task = observer_result(kind, future)

    async def handler(method, params, notification):
        return None

    connection = Connection(handler, source, observers=[lambda event: result], listening=False)
    try:
        error = None
        try:
            await connection.send_notification("example")
        except TypeError as exc:
            error = exc
        assert error is None, "observer Awaitables must be supervised before shutdown"
        if started:
            await asyncio.sleep(0)
        await connection.close()
        # An unstarted custom coroutine adapter has not begun awaiting its
        # underlying Future, so it cannot propagate cancellation to it yet.
        if started or kind == "future":
            assert future.cancelled()
        if task is not None:
            assert task.cancelled()
        assert not connection._tasks._tasks
    finally:
        await connection.close()
        if not future.done():
            future.cancel()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await peer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
async def test_failed_future_uses_observer_error_route_not_delivery_failure(direction: str, caplog) -> None:
    source, peer = memory_transport_pair()
    handled = asyncio.Event()
    future = asyncio.get_running_loop().create_future()
    future.set_exception(RuntimeError("synthetic observer Future failure"))

    async def handler(method, params, notification):
        handled.set()

    connection = Connection(handler, source, observers=[lambda event: future], listening=direction == "incoming")
    try:
        error = None
        if direction == "outgoing":
            try:
                await connection.send_notification("example")
            except TypeError as exc:
                error = exc
            assert error is None, "a failed observer Future must use the observer error route"
        else:
            await peer.send({"jsonrpc": "2.0", "method": "example"})
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(handled.wait(), 0.1)
            assert handled.is_set(), "observer Future errors must not disconnect before dispatch"
        # Let the task and its done callback run before checking the log.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert "Stream observer coroutine failed" in caplog.text
        assert "synthetic observer Future failure" in caplog.text
    finally:
        await connection.close()
        future.exception()  # Also consume it on the broken baseline.
        await peer.close()


@pytest.mark.asyncio
async def test_immediate_close_closes_unstarted_coroutine_observer() -> None:
    source, peer = memory_transport_pair()

    async def handler(method, params, notification):
        return None

    async def observe():
        await asyncio.sleep(10)

    result = observe()
    connection = Connection(handler, source, observers=[lambda event: result], listening=False)
    try:
        await connection.send_notification("example")
        await connection.close()
        assert result.cr_frame is None, "an unstarted observer coroutine must be closed, not leaked"
    finally:
        result.close()
        await connection.close()
        await peer.close()


@pytest.mark.asyncio
async def test_multiple_observer_results_are_bound_independently() -> None:
    source, peer = memory_transport_pair()
    first, second = asyncio.Event(), asyncio.Event()

    async def handler(method, params, notification):
        return None

    async def observe(signal):
        signal.set()

    connection = Connection(
        handler,
        source,
        observers=[lambda event: observe(first), lambda event: observe(second)],
        listening=False,
    )
    try:
        await connection.send_notification("example")
        await asyncio.sleep(0)
        assert first.is_set() and second.is_set()
    finally:
        await connection.close()
        await peer.close()


@pytest.mark.asyncio
async def test_supervisor_future_identity_naming_and_error_route() -> None:
    from acp.task import TaskSupervisor

    supervisor = TaskSupervisor(source="observer-test")
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    errors = []
    tracked = supervisor.create(future, name="future-name", on_error=lambda task, exc: errors.append((task, exc)))
    assert isinstance(tracked, asyncio.Task)
    assert tracked is not future
    failure = RuntimeError("synthetic future failure")
    future.set_exception(failure)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert errors == [(tracked, failure)]

    async def work():
        await asyncio.sleep(10)

    task = asyncio.create_task(work())
    adapter = supervisor.create(task, name="observer-task")
    assert adapter is not task
    assert adapter.get_name() == "observer-task"
    assert task.get_name() != "observer-task"
    await supervisor.shutdown()
    assert task.cancelled()
