"""Closing a connection from its own task must not cancel/gather that task."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("caller", ["notification", "request", "observer", "supervisor"])
@pytest.mark.parametrize("siblings", [False, True])
def test_shutdown_from_owned_task(caller: str, siblings: bool) -> None:
    # A regression can poison cancellation cleanup with a self-await cycle.
    # Run the real asyncio loop in a bounded child, not pytest's own loop.
    source = r"""
import asyncio
from acp.connection import Connection
from acp._transport import memory_transport_pair
from acp.task import TaskSupervisor

async def main():
    finished = asyncio.Event()
    sibling_started = asyncio.Event()
    sibling_stopped = asyncio.Event()
    supervisor = TaskSupervisor(source='test')
    async def sibling():
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_stopped.set()
    if CALLER == 'supervisor':
        async def owned():
            if SIBLINGS:
                supervisor.create(sibling())
                await sibling_started.wait()
            await supervisor.shutdown()
            await asyncio.sleep(0)
            assert not asyncio.current_task().cancelled()
            if SIBLINGS:
                assert sibling_stopped.is_set()
            finished.set()
        task = supervisor.create(owned())
        await asyncio.wait_for(finished.wait(), 2)
        await task
        await supervisor.shutdown()
        assert not supervisor._tasks
        print('closed cleanly')
        return
    local, peer = memory_transport_pair()
    async def handler(method, params, is_notification):
        if CALLER == 'observer':
            return None
        await close_here()
    async def observer(event):
        if CALLER == 'observer':
            await close_here()
    connection = Connection(handler, local, observers=[observer] if CALLER == 'observer' else [])
    async def close_here():
        if SIBLINGS:
            connection._tasks.create(sibling())
            await sibling_started.wait()
        await connection.close()
        await asyncio.sleep(0)
        assert not asyncio.current_task().cancelled()
        if SIBLINGS:
            assert sibling_stopped.is_set()
        finished.set()
    message = {'jsonrpc': '2.0', 'method': 'stop'}
    if CALLER == 'request':
        message['id'] = 1
    await peer.send(message)
    await asyncio.wait_for(finished.wait(), 2)
    await asyncio.sleep(0)
    await connection.close()
    await peer.close()
    assert not connection._tasks._tasks
    print('closed cleanly')
asyncio.run(main())
"""
    source = source.replace("CALLER", repr(caller)).replace("SIBLINGS", repr(siblings))
    result = subprocess.run(  # noqa: S603 - bounded, fixed-source asyncio regression probe
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=8,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "closed cleanly" in result.stdout
    assert "RecursionError" not in result.stderr
    assert "Task was destroyed" not in result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("siblings", [0, 2])
async def test_external_shutdown_still_awaits_owned_task_cleanup(siblings: int) -> None:
    import asyncio

    from acp.task import TaskSupervisor

    supervisor = TaskSupervisor(source="external-test")
    stopped: list[int] = []

    async def owned(index: int) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(index)

    tasks = [supervisor.create(owned(index)) for index in range(siblings)]
    await asyncio.sleep(0)
    await supervisor.shutdown()
    assert sorted(stopped) == list(range(siblings))
    assert all(task.cancelled() for task in tasks)
    assert not supervisor._tasks
    await supervisor.shutdown()
