"""Keep cancellation-cycle regressions isolated so a broken loop cannot hang pytest."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("registrations", [1, 2])
@pytest.mark.parametrize("started", [False, True])
def test_adopted_task_can_close_its_supervisor(nested, registrations, started):
    code = textwrap.dedent(f"""
        import asyncio
        from acp.task import TaskSupervisor

        async def run():
            supervisor = TaskSupervisor(source="reentrant-adapter")
            start = asyncio.Event()
            finished = asyncio.Event()
            async def worker():
                await start.wait()
                await supervisor.shutdown()
                finished.set()
            caller = asyncio.create_task(worker(), name="caller")
            adapters = []
            for _ in range({registrations}):
                adapter = supervisor.create(caller)
                adapters.append(adapter)
                if {nested}:
                    adapters.append(supervisor.create(adapter))
            unrelated = supervisor.create(asyncio.sleep(60))
            if {started}:
                await asyncio.sleep(0)
            start.set()
            await asyncio.wait_for(asyncio.gather(caller, *adapters), 1)
            assert finished.is_set()
            assert not caller.cancelled()
            assert unrelated.cancelled()
            await asyncio.sleep(0)
            assert not supervisor._tasks
            assert not supervisor._adopted
            await supervisor.shutdown()

        asyncio.run(run())
    """)
    result = subprocess.run(  # noqa: S603 - fixed local test program, no external input
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("started", [False, True])
def test_task_observer_can_close_its_connection(started):
    code = textwrap.dedent(f"""
        import asyncio
        from acp._transport import memory_transport_pair
        from acp.connection import Connection

        async def run():
            source, peer = memory_transport_pair()
            start = asyncio.Event()
            async def handler(method, params, notification):
                return None
            async def observer_work():
                await start.wait()
                await connection.close()
            task = asyncio.create_task(observer_work())
            connection = Connection(handler, source, observers=[lambda event: task], listening=False)
            try:
                await connection.send_notification("example")
                assert (await peer.receive())["method"] == "example"
                if {started}:
                    await asyncio.sleep(0)
                start.set()
                await asyncio.wait_for(task, 1)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not task.cancelled()
                assert not connection._tasks._tasks
            finally:
                await connection.close()
                await peer.close()

        asyncio.run(run())
    """)
    result = subprocess.run(  # noqa: S603 - fixed local test program, no external input
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0, result.stdout + result.stderr
