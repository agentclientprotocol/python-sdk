"""A Future-returning observer must not interrupt notification delivery."""

from __future__ import annotations

import asyncio

import pytest

from acp._transport import memory_transport_pair
from acp.connection import Connection


@pytest.mark.asyncio
async def test_future_returning_observer_preserves_delivery() -> None:
    source, peer = memory_transport_pair()
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    observed = []

    def observer(event):
        observed.append(event.direction.value)
        return future

    async def handler(method, params, notification):
        return None

    connection = Connection(handler, source, observers=[observer], listening=False)
    try:
        await connection.send_notification("example", {"value": 1})
        message = await asyncio.wait_for(peer.receive(), 1)
        assert message["method"] == "example"
        assert observed == ["outgoing"]
        future.set_result(None)
        await asyncio.sleep(0)
    finally:
        await connection.close()
        await peer.close()
        if not future.done():
            future.cancel()
