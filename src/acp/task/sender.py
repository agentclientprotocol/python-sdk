from __future__ import annotations

import asyncio
import json
from typing import Any

__all__ = ["MessageSender"]


class MessageSender:
    """Serialise NDJSON frames onto a stream writer.

    ``StreamWriter.write`` buffers a whole frame synchronously, so frames never interleave; the
    lock only serialises ``drain()`` so backpressure is applied in FIFO order. Writes happen in
    the caller's task, so errors and cancellation surface directly in ``send()``.
    """

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self._lock = asyncio.Lock()
        self._closed = False
        self._failure: BaseException | None = None

    async def send(self, payload: dict[str, Any]) -> None:
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        async with self._lock:
            if self._closed:
                raise ConnectionError("Sender closed") from self._failure
            try:
                self._writer.write(data)
                await self._writer.drain()
            except Exception as exc:
                # The stream is unusable after a failed write; fail later sends explicitly
                # instead of silently writing into a closed transport.
                self._failure = exc
                self._closed = True
                raise

    async def close(self) -> None:
        self._closed = True
