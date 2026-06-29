from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from acp.core import run_agent


@pytest.mark.asyncio
async def test_run_agent_closes_connection_when_cancelled(server, agent) -> None:
    sender_created = asyncio.Event()
    sender_closed = asyncio.Event()
    dispatcher_started = asyncio.Event()
    dispatcher_stopped = asyncio.Event()

    class TrackingSender:
        def __init__(self, writer: asyncio.StreamWriter, supervisor: Any) -> None:
            sender_created.set()

        async def send(self, payload: dict[str, Any]) -> None:
            msg = "test does not send messages"
            raise AssertionError(msg)

        async def close(self) -> None:
            sender_closed.set()

    class TrackingDispatcher:
        def start(self) -> None:
            dispatcher_started.set()

        async def stop(self) -> None:
            dispatcher_stopped.set()

    task = asyncio.create_task(
        run_agent(
            agent,
            server.server_writer,
            server.server_reader,
            sender_factory=TrackingSender,
            dispatcher_factory=lambda *args: TrackingDispatcher(),
        )
    )

    await asyncio.wait_for(sender_created.wait(), timeout=1)
    await asyncio.wait_for(dispatcher_started.wait(), timeout=1)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    await asyncio.wait_for(dispatcher_stopped.wait(), timeout=1)
    await asyncio.wait_for(sender_closed.wait(), timeout=1)
