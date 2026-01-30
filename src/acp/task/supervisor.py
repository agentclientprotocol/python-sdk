from __future__ import annotations
from ..py38_compatibility import *

import asyncio
import logging
from typing import Awaitable, Callable, Any
from contextlib import suppress

__all__ = ["TaskSupervisor"]

# Use Any for ErrorHandler to avoid evaluation issues on Python 3.8
ErrorHandler = Any


class TaskSupervisor:
    """Track background tasks and provide graceful shutdown semantics.

    Inspired by fasta2a's task manager, this supervisor keeps a registry of
    asyncio tasks created for request handling so they can be cancelled and
    awaited reliably when the connection closes.
    """

    def __init__(self, *, source: str) -> None:
        self._source = source
        self._tasks: set[Task[Any]] = builtins.set()
        self._closed = False
        self._error_handlers: list[ErrorHandler] = builtins.list()

    def add_error_handler(self, handler: ErrorHandler) -> None:
        self._error_handlers.append(handler)

    def create(
        self,
        coroutine: Awaitable[Any],
        *,
        name: Optional[str] = None,
        on_error: Optional[ErrorHandler] = None,
    ) -> Task[Any]:
        if self._closed:
            msg = f"TaskSupervisor for {self._source} already closed"
            raise RuntimeError(msg)
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(lambda t: self._on_done(t, on_error))
        return task

    def _on_done(self, task: Task[Any], on_error: Optional[ErrorHandler]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            handled = False
            if on_error is not None:
                try:
                    on_error(task, exc)
                    handled = True
                except Exception:
                    logging.exception("Error in %s task-specific error handler", self._source)
            if not handled:
                for handler in self._error_handlers:
                    try:
                        handler(task, exc)
                        handled = True
                    except Exception:
                        logging.exception("Error in %s supervisor error handler", self._source)
            if not handled:
                logging.exception("Unhandled error in %s task", self._source)

    async def shutdown(self) -> None:
        self._closed = True
        if not self._tasks:
            return
        tasks = builtins.list(self._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
