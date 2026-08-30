from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from . import schema

if TYPE_CHECKING:
    from .client import ClientSideConnection
    from .interfaces import Client

__all__ = ["ActiveSession", "SessionMessage", "SessionStop", "SessionUpdate"]


@dataclass(frozen=True, slots=True)
class SessionUpdate:
    notification: schema.UpdateSessionNotification

    @property
    def update(self) -> Any:
        return self.notification.update


@dataclass(frozen=True, slots=True)
class SessionStop:
    notification: schema.UpdateSessionNotification

    @property
    def update(self) -> schema.IdleSessionStateUpdate:
        return cast(schema.IdleSessionStateUpdate, self.notification.update)

    @property
    def stop_reason(self) -> Any:
        return self.update.stop_reason


SessionMessage = SessionUpdate | SessionStop
UpdateHandler = Callable[[schema.UpdateSessionNotification], None]


class SessionUpdateBroker:
    def __init__(self, client: Client) -> None:
        self._client = client
        self._handlers: dict[str, set[UpdateHandler]] = defaultdict(set)
        self._pending: dict[str, list[schema.UpdateSessionNotification]] = defaultdict(list)
        self._captures = 0

    def begin_capture(self) -> None:
        self._captures += 1

    def end_capture(self) -> None:
        self._captures -= 1
        if self._captures == 0:
            self._pending.clear()

    def register(self, session_id: str, handler: UpdateHandler) -> Callable[[], None]:
        self._handlers[session_id].add(handler)
        for notification in self._pending.pop(session_id, []):
            handler(notification)

        def unregister() -> None:
            handlers = self._handlers.get(session_id)
            if handlers is None:
                return
            handlers.discard(handler)
            if not handlers:
                self._handlers.pop(session_id, None)

        return unregister

    async def session_update(self, notification: schema.UpdateSessionNotification) -> None:
        handlers = tuple(self._handlers.get(notification.session_id, ()))
        if handlers:
            for handler in handlers:
                handler(notification)
        elif self._captures:
            self._pending[notification.session_id].append(notification)

        user_handler = getattr(self._client, "session_update", None)
        if user_handler is not None:
            await user_handler(notification)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class ActiveSession:
    """Route one session's updates and track the split v2 prompt lifecycle."""

    def __init__(
        self,
        connection: ClientSideConnection,
        response: schema.NewSessionResponse,
        broker: SessionUpdateBroker,
    ) -> None:
        self._connection = connection
        self._response = response
        self._updates: asyncio.Queue[schema.UpdateSessionNotification] = asyncio.Queue()
        self._unregister = broker.register(response.session_id, self._updates.put_nowait)
        self._prompt_active = False
        self._observed_running = False
        self._disposed = False

    @property
    def session_id(self) -> str:
        return self._response.session_id

    @property
    def new_session_response(self) -> schema.NewSessionResponse:
        return self._response

    async def prompt(self, request: schema.PromptRequest) -> schema.PromptResponse:
        if self._disposed:
            raise RuntimeError("ActiveSession has been disposed")
        if request.session_id != self.session_id:
            raise ValueError(f"Prompt belongs to session {request.session_id!r}, not {self.session_id!r}")
        if self._prompt_active:
            raise RuntimeError("Wait for the current prompt to become idle before sending another")

        self._prompt_active = True
        self._observed_running = False
        try:
            return await self._connection.prompt(request)
        except BaseException:
            self._prompt_active = False
            raise

    async def next_update(self) -> SessionMessage:
        if self._disposed:
            raise RuntimeError("ActiveSession has been disposed")
        notification = await self._updates.get()
        update = notification.update
        if self._prompt_active and isinstance(update, schema.RunningSessionStateUpdate):
            self._observed_running = True
        elif self._prompt_active and self._observed_running and isinstance(update, schema.IdleSessionStateUpdate):
            self._prompt_active = False
            self._observed_running = False
            return SessionStop(notification)
        return SessionUpdate(notification)

    async def wait_for_idle(self) -> SessionStop:
        while True:
            message = await self.next_update()
            if isinstance(message, SessionStop):
                return message

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._unregister()

    async def __aenter__(self) -> ActiveSession:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.dispose()
