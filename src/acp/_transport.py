"""Message-level transport seam for :class:`acp.connection.Connection`.

Historically the connection spoke directly to ``asyncio`` byte streams with
newline framing.  To support message-oriented remote transports (Streamable
HTTP + WebSocket) we introduce a small :class:`Transport` protocol that moves
JSON-RPC *messages* (already-decoded ``dict`` payloads) instead of bytes.

The existing stdio path is re-expressed on top of this seam via
:class:`NdjsonTransport`, which wraps the current byte-stream framing so there
is **zero behaviour change** for stdio users.  :func:`memory_transport_pair`
gives two linked in-memory transports, used by the HTTP/WS server to bind an
``AgentSideConnection`` to its message pump.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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
    """Transport backed by newline-delimited JSON over asyncio byte streams.

    This preserves the exact framing, buffer-limit-overrun handling, and receive
    timeout semantics that previously lived inside ``Connection`` so that stdio
    behaviour is byte-for-byte unchanged.
    """

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
                message: dict[str, Any] = json.loads(line)
            except Exception:
                logging.exception("Error parsing JSON-RPC message")
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
    other.  Closing an end enqueues an EOF (``None``) for its peer.  This mirrors
    the ``TransformStream`` pair the TypeScript SDK uses to bind a server-side
    connection to its HTTP/WS message pump.
    """
    a_to_b: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    b_to_a: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    left = _MemoryTransport(outbox=a_to_b, inbox=b_to_a)
    right = _MemoryTransport(outbox=b_to_a, inbox=a_to_b)
    return left, right
