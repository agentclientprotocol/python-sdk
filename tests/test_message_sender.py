from __future__ import annotations

import asyncio

import pytest

from acp.task import MessageSender


class _FakeWriter:
    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self.written: list[bytes] = []
        self.release = asyncio.Event()
        self.block = False

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        if self.block:
            await self.release.wait()
        if self.fail_with is not None:
            raise self.fail_with


def _make_sender(writer: _FakeWriter) -> MessageSender:
    return MessageSender(writer)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_sends_are_written_in_call_order() -> None:
    writer = _FakeWriter()
    writer.block = True
    sender = _make_sender(writer)

    tasks = [asyncio.create_task(sender.send({"n": i})) for i in range(3)]
    await asyncio.sleep(0)
    writer.release.set()
    await asyncio.wait_for(asyncio.gather(*tasks), 1)
    assert writer.written == [b'{"n":0}\n', b'{"n":1}\n', b'{"n":2}\n']


@pytest.mark.asyncio
async def test_send_after_close_raises() -> None:
    writer = _FakeWriter()
    sender = _make_sender(writer)
    await sender.send({"a": 1})
    await sender.close()
    with pytest.raises(ConnectionError):
        await asyncio.wait_for(sender.send({"a": 2}), 1)
    assert writer.written == [b'{"a":1}\n']


@pytest.mark.asyncio
async def test_write_failure_fails_waiting_and_later_sends() -> None:
    writer = _FakeWriter(fail_with=BrokenPipeError("boom"))
    writer.block = True
    sender = _make_sender(writer)

    first = asyncio.create_task(sender.send({"n": 1}))
    waiting = asyncio.create_task(sender.send({"n": 2}))
    await asyncio.sleep(0)
    writer.release.set()

    with pytest.raises(BrokenPipeError):
        await asyncio.wait_for(first, 1)
    with pytest.raises(ConnectionError) as excinfo:
        await asyncio.wait_for(waiting, 1)
    assert isinstance(excinfo.value.__cause__, BrokenPipeError)
    with pytest.raises(ConnectionError):
        await asyncio.wait_for(sender.send({"n": 3}), 1)
    assert len(writer.written) == 1


@pytest.mark.asyncio
async def test_cancelled_send_releases_lock() -> None:
    writer = _FakeWriter()
    writer.block = True
    sender = _make_sender(writer)

    stuck = asyncio.create_task(sender.send({"n": 1}))
    waiting = asyncio.create_task(sender.send({"n": 2}))
    await asyncio.sleep(0)
    stuck.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stuck

    writer.release.set()
    await asyncio.wait_for(waiting, 1)
    assert writer.written == [b'{"n":1}\n', b'{"n":2}\n']
