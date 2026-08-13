from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from acp.core import run_agent


@pytest.mark.asyncio
async def test_run_agent_closes_connection_when_cancelled(agent) -> None:
    receive_started = asyncio.Event()
    transport_closed = asyncio.Event()

    class TrackingTransport:
        async def receive(self) -> dict[str, Any] | None:
            receive_started.set()
            await asyncio.Event().wait()
            return None

        async def send(self, message: dict[str, Any]) -> None:
            msg = "test does not send messages"
            raise AssertionError(msg)

        async def close(self) -> None:
            transport_closed.set()

    task = asyncio.create_task(
        run_agent(
            agent,
            TrackingTransport(),
        )
    )

    await asyncio.wait_for(receive_started.wait(), timeout=1)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    await asyncio.wait_for(transport_closed.wait(), timeout=1)
