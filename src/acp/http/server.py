"""Streamable HTTP connection management and ACP message routing.

:class:`AcpServer` owns a dictionary of active HTTP connections. For each
``initialize`` POST it mints a connection, binds an ``AgentSideConnection`` to an
HTTP transport, and returns an ``Acp-Connection-Id``. Subsequent
server→client messages produced by the agent are fanned out to the correct SSE
stream (connection-scoped or session-scoped) based on their ``sessionId`` /
correlated request id.

HTTP handlers return Starlette responses directly:

* :meth:`AcpServer.handle_post` — returns a Starlette response.
* :meth:`AcpServer.open_stream` — returns an async byte iterator of SSE frames.
* :meth:`AcpServer.handle_delete` — terminates a connection.

The Starlette application in :mod:`acp.http.asgi` supplies requests and streams.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

try:
    from starlette.responses import JSONResponse, Response
except ImportError as exc:
    raise ImportError("The HTTP server requires the 'http' extra: pip install agent-client-protocol[http]") from exc

from .._sse import serialize_sse_event, serialize_sse_keepalive
from ..agent.connection import AgentSideConnection
from .protocol import (
    CONNECTION_ID_HEADER,
    is_initialize_request,
    is_response_message,
    message_id_key,
    method_requires_session_header,
    session_id_from_params,
    session_id_from_result,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ..interfaces import Agent

__all__ = [
    "AcpServer",
    "AgentFactory",
    "OutboundStream",
]

AgentFactory = Callable[[AgentSideConnection], "Agent"]

# How long an idle SSE stream waits before emitting a keepalive comment. Kept in
# sync with the TypeScript reference (15s) so intermediaries do not time out an
# otherwise-healthy but quiet stream.
SSE_KEEPALIVE_INTERVAL_SECONDS = 15.0

# How long ``initialize`` waits for the agent's response before giving up. The
# response is returned synchronously in the HTTP body, so a hung agent must not
# block the POST forever.
INITIALIZE_TIMEOUT_SECONDS = 30.0


class OutboundStream:
    """A backpressure-aware buffer for server→client messages.

    Messages pushed before a subscriber attaches are buffered (bounded) and
    replayed when :meth:`iterate` is first awaited. When the buffer is full,
    :meth:`push` *awaits* until the consumer drains rather than dropping the
    message — dropping a JSON-RPC response would permanently hang the peer's
    pending request. Awaiting propagates backpressure up to the agent's message
    handlers, mirroring the ``ReadableStream`` backpressure in the TypeScript SDK.
    """

    def __init__(self, *, capacity: int = 1024) -> None:
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=capacity)
        self._closed = asyncio.Event()

    async def push(self, message: dict[str, Any]) -> None:
        if self._closed.is_set():
            return
        putter = asyncio.ensure_future(self._queue.put(message))
        closed = asyncio.ensure_future(self._closed.wait())
        try:
            await asyncio.wait({putter, closed}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            # If the stream closed while we were blocked on a full queue, abandon
            # the put; otherwise ensure the close-waiter task is cleaned up.
            for task in (putter, closed):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        # Guarantee the consumer observes EOF even if the buffer is full: make
        # room for the sentinel by dropping one buffered (tail) message, which is
        # acceptable during teardown.
        while not self._try_put_sentinel():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()

    def _try_put_sentinel(self) -> bool:
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            return False
        return True

    async def iterate(self) -> AsyncGenerator[dict[str, Any], None]:
        while True:
            message = await self._queue.get()
            if message is None:
                return
            yield message


class _HttpTransport:
    """Receive POST messages and route agent output directly to HTTP/SSE.

    Only initialize returns in a POST body. New-session responses use the
    connection stream; later responses follow their request's session route.
    Requests and notifications from the agent carry their own sessionId.
    """

    def __init__(self, initialize_id: Any) -> None:
        self._initialize_id = message_id_key(initialize_id)
        self.initialize_response: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._incoming: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._closed = False
        self.connection_stream = OutboundStream()
        self.session_streams: dict[str, OutboundStream] = {}
        self._pending_routes: dict[str, str] = {}

    async def receive(self) -> dict[str, Any] | None:
        return await self._incoming.get()

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise ConnectionError("Transport closed")
        session_id = session_id_from_params(message.get("params"))
        if is_response_message(message):
            key = message_id_key(message.get("id"))
            if key == self._initialize_id and not self.initialize_response.done():
                self.initialize_response.set_result(message)
                return
            session_id = self._pending_routes.pop(key, None) if key is not None else None
            established = session_id_from_result(message.get("result"))
            if established is not None:
                self.session_streams.setdefault(established, OutboundStream())
                # The client must learn the session ID before opening its stream.
                session_id = None
        stream = (
            self.session_streams.get(session_id, self.connection_stream)
            if session_id is not None
            else self.connection_stream
        )
        await stream.push(message)

    async def deliver_to_agent(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise ConnectionError("Transport closed")
        if "id" in message and "method" in message:
            session_id = session_id_from_params(message.get("params"))
            key = message_id_key(message["id"])
            if session_id is not None and key is not None:
                self._pending_routes[key] = session_id
        self._incoming.put_nowait(dict(message))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._incoming.put_nowait(None)
        self.initialize_response.cancel()
        self._pending_routes.clear()
        self.connection_stream.close()
        for stream in self.session_streams.values():
            stream.close()


class AcpServer:
    """Manage ACP HTTP connections and return Starlette responses.

    Args:
        agent_factory: Called once per connection with the bound
            ``AgentSideConnection`` to produce a per-connection ``Agent``.
    """

    def __init__(self, agent_factory: AgentFactory) -> None:
        self.agent_factory = agent_factory
        self._connections: dict[str, tuple[AgentSideConnection, _HttpTransport]] = {}

    # -- POST ---------------------------------------------------------------

    async def handle_post(
        self,
        message: Any,
        *,
        content_type: str | None,
        connection_id: str | None,
        session_id: str | None,
    ) -> Response:
        if content_type is None or not content_type.lower().startswith("application/json"):
            return JSONResponse({"error": "Content-Type must be application/json"}, status_code=415)
        if isinstance(message, list):
            return JSONResponse({"error": "Batch requests are not supported"}, status_code=501)
        if not isinstance(message, dict):
            return JSONResponse({"error": "Invalid JSON-RPC message"}, status_code=400)

        if is_initialize_request(message):
            return await self._handle_initialize(message)

        if connection_id is None:
            return JSONResponse({"error": "Missing connection id"}, status_code=400)
        connection = self._connections.get(connection_id)
        if connection is None:
            return JSONResponse({"error": "Unknown connection id"}, status_code=404)
        _, transport = connection

        method = message.get("method")
        if method_requires_session_header(method) and session_id is None:
            return JSONResponse({"error": "Missing session id header"}, status_code=400)
        if session_id is not None and session_id not in transport.session_streams:
            # A session-scoped POST references an unknown session.
            return JSONResponse({"error": "Unknown session id"}, status_code=404)

        await transport.deliver_to_agent(message)
        return Response(status_code=202)

    async def _handle_initialize(self, message: dict[str, Any]) -> Response:
        connection_id = uuid.uuid4().hex
        transport = _HttpTransport(message.get("id"))
        conn = AgentSideConnection(self.agent_factory, transport)
        self._connections[connection_id] = (conn, transport)
        # Deliver initialize to the agent and await its response so we can return
        # the 200 body synchronously (initialize is the one blocking POST). If the
        # agent never responds (timeout) or errors, tear the just-created
        # connection down instead of leaking its agent connection.
        try:
            await transport.deliver_to_agent(message)
            response = await asyncio.wait_for(transport.initialize_response, timeout=INITIALIZE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            await self.handle_delete(connection_id=connection_id)
            return JSONResponse({"error": "initialize timed out"}, status_code=504)
        except Exception:
            await self.handle_delete(connection_id=connection_id)
            return JSONResponse({"error": "initialize failed"}, status_code=500)
        except asyncio.CancelledError:
            await self.handle_delete(connection_id=connection_id)
            raise
        return JSONResponse(response, headers={CONNECTION_ID_HEADER: connection_id})

    # -- GET / SSE ----------------------------------------------------------

    def validate_stream(self, *, connection_id: str | None, session_id: str | None) -> Response | None:
        """Validate a GET SSE request. Returns an error response, or None if OK."""
        if connection_id is None:
            return JSONResponse({"error": "Missing connection id"}, status_code=400)
        connection = self._connections.get(connection_id)
        if connection is None:
            return JSONResponse({"error": "Unknown connection id"}, status_code=404)
        _, transport = connection
        if session_id is not None and session_id not in transport.session_streams:
            return JSONResponse({"error": "Unknown session id"}, status_code=404)
        return None

    async def open_stream(
        self,
        *,
        connection_id: str,
        session_id: str | None,
    ) -> AsyncGenerator[bytes, None]:
        """Yield SSE byte frames for a connection- or session-scoped stream.

        Emits a keepalive comment whenever the stream is idle for longer than
        :data:`SSE_KEEPALIVE_INTERVAL_SECONDS` so that idle-timeout intermediaries
        (proxies, load balancers) do not close an otherwise-healthy stream.
        """
        connection = self._connections.get(connection_id)
        if connection is None:
            return
        _, transport = connection
        stream = transport.session_streams[session_id] if session_id is not None else transport.connection_stream
        messages = stream.iterate()
        pending: asyncio.Task[dict[str, Any]] | None = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.ensure_future(messages.__anext__())
                done, _ = await asyncio.wait({pending}, timeout=SSE_KEEPALIVE_INTERVAL_SECONDS)
                if not done:
                    # Idle: emit a keepalive and keep awaiting the same message.
                    yield serialize_sse_keepalive()
                    continue
                try:
                    message = pending.result()
                except StopAsyncIteration:
                    return
                finally:
                    pending = None
                yield serialize_sse_event(message)
        finally:
            if pending is not None:
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pending
            await messages.aclose()

    # -- DELETE -------------------------------------------------------------

    async def handle_delete(self, *, connection_id: str | None) -> Response:
        if connection_id is None:
            return JSONResponse({"error": "Missing connection id"}, status_code=400)
        connection = self._connections.pop(connection_id, None)
        if connection is None:
            return JSONResponse({"error": "Unknown connection id"}, status_code=404)
        conn, _ = connection
        await conn.close()
        return Response(status_code=202)

    async def close(self) -> None:
        for connection_id in list(self._connections):
            await self.handle_delete(connection_id=connection_id)
