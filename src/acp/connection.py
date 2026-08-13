from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from ._transport import NdjsonTransport, Transport
from .exceptions import RequestError
from .task import MessageSender, TaskSupervisor
from .telemetry import span_context

JsonValue = Any
MethodHandler = Callable[[str, JsonValue | None, bool], Awaitable[JsonValue | None]]


__all__ = ["Connection", "JsonValue", "MethodHandler", "StreamDirection", "StreamEvent"]


class StreamDirection(str, Enum):
    INCOMING = "incoming"
    OUTGOING = "outgoing"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    direction: StreamDirection
    message: dict[str, Any]


StreamObserver = Callable[[StreamEvent], Awaitable[None] | None]


class Connection:
    """Minimal JSON-RPC 2.0 connection over newline-delimited JSON frames."""

    def __init__(
        self,
        handler: MethodHandler,
        writer: asyncio.StreamWriter | Transport,
        reader: asyncio.StreamReader | None = None,
        *,
        observers: list[StreamObserver] | None = None,
        listening: bool = True,
        receive_timeout: float | None = None,
    ) -> None:
        self._handler = handler
        self._next_request_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._tasks = TaskSupervisor(source="acp.Connection")
        self._tasks.add_error_handler(self._on_task_error)
        self._closed = False
        self._disconnected = False
        # Two construction forms:
        #   * message-level: ``Connection(handler, transport)`` (reader omitted)
        #   * byte-level:    ``Connection(handler, writer, reader)`` (stdio path)
        # We discriminate on ``reader`` rather than ``isinstance(writer, Transport)``
        # because a runtime-checkable Protocol would spuriously match duck-typed
        # test doubles (e.g. ``MagicMock``).
        if reader is None:
            self._transport: Transport = cast("Transport", writer)
        else:
            sender = MessageSender(cast("asyncio.StreamWriter", writer), self._tasks)
            self._transport = NdjsonTransport(reader, sender, receive_timeout=receive_timeout)
        self._observers: list[StreamObserver] = list(observers or [])
        if listening:
            self._recv_task = self._tasks.create(
                self._receive_loop(),
                name="acp.Connection.receive",
                on_error=self._on_receive_error,
            )
        else:
            self._recv_task = None

    async def close(self) -> None:
        """Stop the receive loop and cancel any in-flight handler tasks."""
        if self._closed:
            return
        self._closed = True
        self._reject_all_outgoing(ConnectionError("Connection closed"))
        try:
            await self._transport.close()
        finally:
            await self._tasks.shutdown()

    async def main_loop(self) -> None:
        try:
            await self._receive_loop()
        except Exception as exc:
            logging.exception("Connection main loop failed", exc_info=exc)
            self._on_receive_error(None, exc)  # type: ignore[arg-type]
            raise

    async def __aenter__(self) -> Connection:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def add_observer(self, observer: StreamObserver) -> None:
        """Register a callback that receives every raw JSON-RPC message."""
        self._observers.append(observer)

    async def send_request(self, method: str, params: JsonValue | None = None) -> Any:
        self._raise_if_unavailable()
        request_id = self._next_request_id
        self._next_request_id += 1
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        try:
            await self._transport.send(payload)
        except BaseException:
            self._pending.pop(request_id, None)
            future.cancel()
            raise
        self._notify_observers(StreamDirection.OUTGOING, payload)
        try:
            return await future
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            future.cancel()
            raise

    async def send_notification(self, method: str, params: JsonValue | None = None) -> None:
        self._raise_if_unavailable()
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        await self._transport.send(payload)
        self._notify_observers(StreamDirection.OUTGOING, payload)

    async def _receive_loop(self) -> None:
        try:
            while True:
                message = await self._transport.receive()
                if message is None:
                    break
                self._notify_observers(StreamDirection.INCOMING, message)
                self._process_message(message)
        except asyncio.CancelledError:
            return
        except asyncio.TimeoutError:
            raise RequestError.internal_error({"details": "Agent timeout"}) from None
        self._disconnect()

    def _process_message(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        has_id = "id" in message
        if method is not None:  # this is a request or notification
            # {"jsonrpc": "2.0", "id": 1, "method": "foo", "params": {...}}  # request
            # {"jsonrpc": "2.0", "method": "foo", "params: {...}}  # notification
            self._tasks.create(
                self._run_request(message) if has_id else self._run_notification(message),
                name="acp.Connection.request" if has_id else "acp.Connection.notification",
            )
            return
        if has_id:  # this is a response, {"id", "result" | "error"}
            self._handle_response(message)

    def _notify_observers(self, direction: StreamDirection, message: dict[str, Any]) -> None:
        if not self._observers:
            return
        snapshot = copy.deepcopy(message)
        event = StreamEvent(direction, snapshot)
        for observer in list(self._observers):
            try:
                result = observer(event)
            except Exception:
                logging.exception("Stream observer failed", exc_info=True)
                continue
            if inspect.isawaitable(result):
                self._tasks.create(
                    result,
                    name=f"acp.Connection.observer.{direction.value}",
                    on_error=self._on_observer_error,
                )

    def _on_observer_error(self, task: asyncio.Task[Any], exc: BaseException) -> None:
        logging.exception("Stream observer coroutine failed", exc_info=exc)

    async def _run_request(self, message: dict[str, Any]) -> None:
        payload = await self._execute_request(message)
        await self._transport.send(payload)
        self._notify_observers(StreamDirection.OUTGOING, payload)

    async def _execute_request(self, message: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        method = message["method"]
        with span_context(
            "acp.request",
            attributes={"method": method},
        ):
            try:
                result = await self._handler(method, message.get("params"), False)
                if isinstance(result, BaseModel):
                    result = result.model_dump(
                        mode="json",
                        by_alias=True,
                        exclude_none=True,
                        exclude_unset=True,
                    )
                payload["result"] = result if result is not None else None
            except RequestError as exc:
                payload["error"] = exc.to_error_obj()
            except ValidationError as exc:
                payload["error"] = RequestError.invalid_params({"errors": exc.errors()}).to_error_obj()
            except Exception as exc:
                logging.exception(
                    "Unhandled error while handling request method=%s",
                    method,
                    exc_info=exc,
                )
                try:
                    data = json.loads(str(exc))
                except Exception:
                    data = {"details": str(exc)}
                payload["error"] = RequestError.internal_error(data).to_error_obj()
        return payload

    async def _run_notification(self, message: dict[str, Any]) -> None:
        method = message["method"]
        with span_context("acp.notification", attributes={"method": method}):
            try:
                await self._handler(method, message.get("params"), True)
            except Exception as exc:
                logging.exception(
                    "Unhandled error while handling notification method=%s",
                    method,
                    exc_info=exc,
                )

    def _handle_response(self, message: dict[str, Any]) -> None:
        request_id = message["id"]
        future = self._pending.pop(request_id, None)
        if future is None or future.done():
            return
        if "result" in message:
            future.set_result(message.get("result"))
            return
        if "error" in message:
            error_obj = message.get("error") or {}
            future.set_exception(
                RequestError(error_obj.get("code", -32603), error_obj.get("message", "Error"), error_obj.get("data"))
            )
            return
        future.set_result(None)

    def _on_receive_error(self, task: asyncio.Task[Any], exc: BaseException) -> None:
        logging.exception("Receive loop failed", exc_info=exc)
        self._disconnect()

    def _on_task_error(self, task: asyncio.Task[Any], exc: BaseException) -> None:
        logging.exception("Background task failed", exc_info=exc)

    def _disconnect(self) -> None:
        if self._disconnected:
            return
        self._disconnected = True
        self._reject_all_outgoing(ConnectionError("Connection closed"))

    def _reject_all_outgoing(self, error: BaseException) -> None:
        pending = list(self._pending.values())
        self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(error)

    def _raise_if_unavailable(self) -> None:
        if self._disconnected or self._closed:
            raise ConnectionError("Connection closed")
