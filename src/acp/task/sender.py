from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from .supervisor import TaskSupervisor

__all__ = ["MessageSender"]


@dataclass(slots=True)
class _PendingSend:
    payload: bytes
    future: asyncio.Future[None]


class MessageSender:
    def __init__(self, writer: asyncio.StreamWriter, supervisor: TaskSupervisor) -> None:
        self._writer = writer
        self._queue: asyncio.Queue[_PendingSend | None] = asyncio.Queue()
        self._closed = False
        self._active: _PendingSend | None = None
        self._terminal_error: BaseException | None = None
        self._terminal_cancelled = False
        self._task = supervisor.create(self._loop(), name="acp.Sender.loop", on_error=self._on_error)
        self._task.add_done_callback(self._on_task_done)

    async def send(self, payload: dict[str, Any]) -> None:
        if self._terminal_error is not None:
            if self._terminal_cancelled:
                raise asyncio.CancelledError
            error = self._terminal_error
            raise error
        if self._closed:
            raise asyncio.CancelledError
        if self._task.done():
            if self._task.cancelled():
                self._set_terminal(ConnectionError("Sender closed"), cancel_pending=True)
                raise asyncio.CancelledError
            error = self._task.exception() or ConnectionError("Sender closed")
            self._set_terminal(error)
            raise error
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._queue.put(_PendingSend(data, future))
        await future

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _loop(self) -> None:
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    return
                self._active = item
                try:
                    self._writer.write(item.payload)
                    await self._writer.drain()
                except asyncio.CancelledError:
                    self._set_terminal(ConnectionError("Sender closed"), cancel_pending=True)
                    raise
                except Exception as exc:
                    self._set_terminal(exc)
                    raise
                else:
                    if not item.future.done():
                        item.future.set_result(None)
                finally:
                    self._active = None
        except asyncio.CancelledError:
            self._set_terminal(ConnectionError("Sender closed"), cancel_pending=True)
            raise

    def _set_terminal(self, error: BaseException, *, cancel_pending: bool = False) -> None:
        if self._terminal_error is not None:
            return
        self._terminal_error = error
        self._terminal_cancelled = cancel_pending
        if cancel_pending:
            self._cancel_pending()
            return
        if self._active is not None and not self._active.future.done():
            self._active.future.set_exception(error)
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not None and not item.future.done():
                item.future.set_exception(error)

    def _cancel_pending(self) -> None:
        if self._active is not None and not self._active.future.done():
            self._active.future.cancel()
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not None and not item.future.done():
                item.future.cancel()

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            self._set_terminal(ConnectionError("Sender closed"), cancel_pending=True)
        else:
            error = task.exception()
            if error is not None:
                self._set_terminal(error)

    def _on_error(self, task: asyncio.Task[Any], exc: BaseException) -> None:
        logging.exception("Send loop failed", exc_info=exc)
