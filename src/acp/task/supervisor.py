from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any

__all__ = ["TaskSupervisor"]

ErrorHandler = Callable[[asyncio.Task[Any], BaseException], None]


async def _await_result(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


class TaskSupervisor:
    """Track background tasks and provide graceful shutdown semantics.

    Inspired by fasta2a's task manager, this supervisor keeps a registry of
    asyncio tasks created for request handling so they can be cancelled and
    awaited reliably when the connection closes.
    """

    def __init__(self, *, source: str) -> None:
        self._source = source
        self._tasks: set[asyncio.Task[Any]] = set()
        self._adopted: dict[asyncio.Future[Any], asyncio.Future[Any]] = {}
        self._closed = False
        self._error_handlers: list[ErrorHandler] = []

    def add_error_handler(self, handler: ErrorHandler) -> None:
        self._error_handlers.append(handler)

    def create(
        self,
        coroutine: Awaitable[Any],
        *,
        name: str | None = None,
        on_error: ErrorHandler | None = None,
    ) -> asyncio.Task[Any]:
        """Create a supervised Task, adopting cancellation of its awaitable.

        Existing Futures and Tasks are awaited by a new Task, so their names
        remain unchanged and error handlers always receive a Task. Shutdown
        cancels an adopted Future or Task even before its adapter starts.
        Each registration creates its own Task and completion callback.
        """
        if self._closed:
            msg = f"TaskSupervisor for {self._source} already closed"
            raise RuntimeError(msg)
        loop = asyncio.get_running_loop()
        if not inspect.isawaitable(coroutine):
            raise TypeError("An awaitable is required")
        if isinstance(coroutine, asyncio.Future) and coroutine.get_loop() is not loop:
            raise ValueError("Awaitable belongs to a different loop")
        # Schedule coroutine objects directly: wrapping one would leak it if the
        # adapter were cancelled before ever starting.
        task = loop.create_task(coroutine if inspect.iscoroutine(coroutine) else _await_result(coroutine), name=name)
        if isinstance(coroutine, asyncio.Future):
            self._adopted[task] = coroutine

            def cancel_adopted(done: asyncio.Task[Any]) -> None:
                if not done.cancelled():
                    return
                if not coroutine.done():
                    coroutine.cancel()
                elif not coroutine.cancelled():
                    # The adapter may never have started to consume a failure.
                    coroutine.exception()

            task.add_done_callback(cancel_adopted)
        self._tasks.add(task)
        task.add_done_callback(lambda t: self._on_done(t, on_error))
        return task

    def _on_done(self, task: asyncio.Task[Any], on_error: ErrorHandler | None) -> None:
        self._tasks.discard(task)
        self._adopted.pop(task, None)
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

    def _awaits_task(self, task: asyncio.Future[Any], target: asyncio.Task[Any] | None) -> bool:
        """Follow only explicitly adopted dependencies owned by this supervisor."""
        seen: set[asyncio.Future[Any]] = set()
        while task not in seen:
            if task is target:
                return True
            seen.add(task)
            adopted = self._adopted.get(task)
            if adopted is None:
                return False
            task = adopted
        return False

    async def shutdown(self) -> None:
        self._closed = True
        if not self._tasks:
            return
        # A handler or async observer may close its own connection. Neither the
        # caller nor adapters awaiting it may be cancelled or gathered here:
        # cancelling an adapter propagates cancellation back into the caller.
        current = asyncio.current_task()
        tasks = [task for task in self._tasks if not self._awaits_task(task, current)]
        for task in tasks:
            task.cancel()
        # Failures were already routed to the error handlers by ``_on_done``; shutdown only
        # needs to wait for the tasks to finish, so don't let them escape from here.
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.difference_update(tasks)
