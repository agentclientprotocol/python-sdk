"""Message-level transport seam for :class:`acp.connection.Connection`.

Transports exchange decoded JSON-RPC objects. :class:`NdjsonTransport` provides
newline framing for byte streams, while :func:`memory_transport_pair` connects
two peers in memory.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ._jsonrpc import decode_message, error_response
from .exceptions import RequestError

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from .task import MessageSender

__all__ = [
    "NdjsonTransport",
    "Transport",
    "memory_transport_pair",
]


@runtime_checkable
class Transport(Protocol):
    """A bidirectional stream of JSON-RPC messages.

    Implementations move already-decoded ``dict`` payloads.  ``receive`` returns
    ``None`` to signal end-of-stream (EOF).  ``send`` may raise if the message
    could not be delivered (e.g. an HTTP POST failing before any JSON-RPC
    response exists); callers correlate such failures with pending requests.
    """

    async def send(self, message: dict[str, Any]) -> None: ...

    async def receive(self) -> dict[str, Any] | None: ...

    async def close(self) -> None: ...


class NdjsonTransport:
    """Transport backed by newline-delimited JSON over asyncio byte streams."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        sender: MessageSender,
        *,
        receive_timeout: float | None = None,
    ) -> None:
        self._reader = reader
        self._sender = sender
        self._receive_timeout = receive_timeout
        self._error_send_failed = False

    async def send(self, message: dict[str, Any]) -> None:
        await self._sender.send(message)

    async def receive(self) -> dict[str, Any] | None:
        while True:
            line = await self._read_line()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                message = decode_message(line)
            except RequestError as error:
                if not self._error_send_failed:
                    try:
                        await self.send(error_response(error))
                    except ConnectionError:
                        self._error_send_failed = True
                continue
            if message is None:
                continue
            return message

    async def close(self) -> None:
        await self._sender.close()

    async def _read_line(self) -> bytes:
        chunks: list[bytes] = []
        try:
            while True:
                try:
                    line = await self._wait_for_reader(self._reader.readuntil(b"\n"))
                except asyncio.LimitOverrunError as exc:
                    chunks.append(await self._wait_for_reader(self._reader.readexactly(exc.consumed)))
                else:
                    chunks.append(line)
                    return b"".join(chunks)
        except asyncio.IncompleteReadError as exc:
            chunks.append(exc.partial)
            return b"".join(chunks)

    async def _wait_for_reader(self, awaitable: Awaitable[bytes]) -> bytes:
        return await asyncio.wait_for(awaitable, timeout=self._receive_timeout)


class _MemoryTransport:
    """One end of an in-process :func:`memory_transport_pair`."""

    def __init__(
        self,
        outbox: asyncio.Queue[dict[str, Any] | None],
        inbox: asyncio.Queue[dict[str, Any] | None],
    ) -> None:
        self._outbox = outbox
        self._inbox = inbox
        self._closed = False

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise ConnectionError("Transport closed")
        await self._outbox.put(dict(message))

    async def receive(self) -> dict[str, Any] | None:
        return await self._inbox.get()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._outbox.put_nowait(None)


def memory_transport_pair() -> tuple[Transport, Transport]:
    """Return two linked in-memory transports.

    A message ``send`` on one end becomes available via ``receive`` on the
    other. Closing an end enqueues an EOF (``None``) for its peer.
    """
    a_to_b: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    b_to_a: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    left = _MemoryTransport(outbox=a_to_b, inbox=b_to_a)
    right = _MemoryTransport(outbox=b_to_a, inbox=a_to_b)
    return left, right
