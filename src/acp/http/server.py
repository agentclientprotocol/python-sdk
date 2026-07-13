"""Framework-agnostic Streamable HTTP server core (port of #155 server.ts + connection.ts).

:class:`AcpServer` owns an in-memory :class:`ConnectionRegistry`.  For each
``initialize`` POST it mints a connection, binds an ``AgentSideConnection`` to an
in-memory transport pair, and returns an ``Acp-Connection-Id``.  Subsequent
server→client messages produced by the agent are fanned out to the correct SSE
stream (connection-scoped or session-scoped) based on their ``sessionId`` /
correlated request id.

The core exposes small, transport-neutral entry points:

* :meth:`AcpServer.handle_post` — returns a :class:`PostResult` (status + body).
* :meth:`AcpServer.open_stream` — returns an async byte iterator of SSE frames.
* :meth:`AcpServer.handle_delete` — terminates a connection.

The ASGI adapter in :mod:`acp.http.asgi` maps these onto ASGI messages.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .._sse import serialize_sse_event, serialize_sse_keepalive
from .._transport import memory_transport_pair
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
    from collections.abc import AsyncIterator

    from ..interfaces import Agent

__all__ = [
    "AcpServer",
    "AgentFactory",
    "ConnectionRegistry",
    "ConnectionState",
    "OutboundStream",
    "PostResult",
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


@dataclass
class PostResult:
    """Outcome of a POST request."""

    status: int
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)


class OutboundStream:
    """A backpressure-aware buffer for server→client messages.

    Messages pushed before a subscriber attaches are buffered (bounded) and
    replayed when :meth:`iterate` is first awaited. When the buffer is full,
    :meth:`push` *awaits* until the consumer drains rather than dropping the
    message — dropping a JSON-RPC response would permanently hang the peer's
    pending request. Awaiting propagates backpressure up to the agent's message
    pump, mirroring the ``ReadableStream`` backpressure in the TypeScript SDK.
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

    async def iterate(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            message = await self._queue.get()
            if message is None:
                return
            yield message


class ConnectionState:
    """Owns an ``AgentSideConnection`` bound to an in-memory transport pair.

    The agent writes server→client messages onto the server end of the pair; a
    pump task reads them and routes each to the connection-scoped stream or the
    right session-scoped stream.
    """

    def __init__(self, connection_id: str, agent_factory: AgentFactory, *, multiplex: bool = False) -> None:
        self.connection_id = connection_id
        # ``server_side`` is what the AgentSideConnection talks over; ``pump_side``
        # is what we read agent→client traffic from and inject client→agent on.
        server_side, pump_side = memory_transport_pair()
        self._pump_side = pump_side
        self._agent_conn = AgentSideConnection(agent_factory, server_side, listening=True)
        self.connection_stream = OutboundStream()
        self.session_streams: dict[str, OutboundStream] = {}
        # WebSocket mode: multiplex *all* agent→client traffic onto one stream
        # (the single socket) instead of splitting across SSE streams.
        self._multiplex: OutboundStream | None = OutboundStream() if multiplex else None
        # Maps a request id -> sessionId, so responses to session-scoped client
        # requests route back onto the right session stream.
        self._pending_routes: dict[str, str] = {}
        # Request ids whose response should be captured (e.g. initialize) instead
        # of being pushed to a stream.
        self._response_waiters: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._pump_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._pump_task = asyncio.ensure_future(self._pump())

    async def _pump(self) -> None:
        try:
            while True:
                message = await self._pump_side.receive()
                if message is None:
                    return
                await self._route_outbound(message)
        except asyncio.CancelledError:
            return

    async def _route_outbound(self, message: dict[str, Any]) -> None:
        """Route an agent→client message to the correct SSE stream.

        Rules (matching the RFD):

        * A response to a session-*establishing* request (``session/new`` /
          ``session/load`` — result carries a ``sessionId``) goes on the
          **connection-scoped** stream, because the client does not yet have the
          session-scoped stream open.  We register the session so its stream can
          be opened on the next GET.
        * A response to an already-session-scoped client request routes onto that
          session's stream (looked up via ``_pending_routes`` by request id).
        * A server→client message carrying a ``sessionId`` in params (a
          notification or request) routes onto that session's stream.
        * Everything else goes on the connection-scoped stream.
        """
        if is_response_message(message):
            await self._route_response(message)
            return
        # Requests/notifications: route by sessionId in params if present.
        if self._multiplex is not None:
            await self._multiplex.push(message)
            return
        session_id = session_id_from_params(message.get("params"))
        if session_id is not None and session_id in self.session_streams:
            await self.session_streams[session_id].push(message)
            return
        await self.connection_stream.push(message)

    async def _route_response(self, message: dict[str, Any]) -> None:
        key = message_id_key(message.get("id"))
        # A captured response (e.g. initialize) resolves its waiter instead of
        # being pushed to any stream.
        if key is not None and key in self._response_waiters:
            waiter = self._response_waiters.pop(key)
            if not waiter.done():
                waiter.set_result(message)
            return
        # Register any newly-established session so unknown-session validation
        # succeeds regardless of transport.
        established = session_id_from_result(message.get("result"))
        if established is not None:
            self.ensure_session_stream(established)
        routed = self._pending_routes.pop(key, None) if key is not None else None
        if self._multiplex is not None:
            await self._multiplex.push(message)
            return
        # session/new | session/load results (``established``) go on the
        # connection-scoped stream; already-session-scoped responses route to the
        # session stream recorded when the request came in.
        if established is None and routed is not None and routed in self.session_streams:
            await self.session_streams[routed].push(message)
            return
        await self.connection_stream.push(message)

    async def deliver_to_agent(self, message: dict[str, Any]) -> None:
        """Inject a client→server message into the agent connection."""
        # Track session-scoped client requests so their responses route back.
        if "id" in message and "method" in message:
            session_id = session_id_from_params(message.get("params"))
            if session_id is not None:
                key = message_id_key(message["id"])
                if key is not None:
                    self._pending_routes[key] = session_id
        await self._pump_side.send(message)

    async def request_response(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a request to the agent and await its correlated response.

        Used for the ``initialize`` POST, which is the one request whose response
        is returned synchronously in the HTTP body rather than over an SSE stream.
        """
        key = message_id_key(message.get("id"))
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        if key is not None:
            self._response_waiters[key] = future
        await self.deliver_to_agent(message)
        return await asyncio.wait_for(future, timeout=INITIALIZE_TIMEOUT_SECONDS)

    def ensure_session_stream(self, session_id: str) -> OutboundStream:
        stream = self.session_streams.get(session_id)
        if stream is None:
            stream = OutboundStream()
            self.session_streams[session_id] = stream
        return stream

    def has_session(self, session_id: str) -> bool:
        return session_id in self.session_streams

    async def iter_all_outbound(self) -> AsyncIterator[dict[str, Any]]:
        """Iterate every agent→client message (WebSocket multiplex mode)."""
        if self._multiplex is None:
            msg = "iter_all_outbound requires a multiplex connection (WebSocket)"
            raise RuntimeError(msg)
        async for message in self._multiplex.iterate():
            yield message

    async def close(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump_task
        self.connection_stream.close()
        for stream in self.session_streams.values():
            stream.close()
        if self._multiplex is not None:
            self._multiplex.close()
        with contextlib.suppress(Exception):
            await self._pump_side.close()
        with contextlib.suppress(Exception):
            await self._agent_conn.close()


class ConnectionRegistry:
    """In-memory ``connectionId -> ConnectionState`` registry."""

    def __init__(self) -> None:
        self._connections: dict[str, ConnectionState] = {}

    def create(self, agent_factory: AgentFactory) -> ConnectionState:
        connection_id = uuid.uuid4().hex
        state = ConnectionState(connection_id, agent_factory)
        state.start()
        self._connections[connection_id] = state
        return state

    def create_multiplex(self, agent_factory: AgentFactory) -> ConnectionState:
        """Create a connection whose agent→client traffic is multiplexed onto one
        stream (used by the WebSocket transport)."""
        connection_id = uuid.uuid4().hex
        state = ConnectionState(connection_id, agent_factory, multiplex=True)
        state.start()
        self._connections[connection_id] = state
        return state

    def get(self, connection_id: str) -> ConnectionState | None:
        return self._connections.get(connection_id)

    async def remove(self, connection_id: str) -> None:
        state = self._connections.pop(connection_id, None)
        if state is not None:
            await state.close()

    async def close_all(self) -> None:
        for connection_id in list(self._connections):
            await self.remove(connection_id)


class AcpServer:
    """Framework-agnostic Streamable HTTP + WebSocket server core.

    Args:
        agent_factory: Called once per connection with the bound
            ``AgentSideConnection`` to produce a per-connection ``Agent``.
    """

    def __init__(self, agent_factory: AgentFactory) -> None:
        self._agent_factory = agent_factory
        self._registry = ConnectionRegistry()

    @property
    def registry(self) -> ConnectionRegistry:
        return self._registry

    def create_websocket_connection(self) -> ConnectionState:
        """Create a new multiplexed connection for a WebSocket upgrade."""
        return self._registry.create_multiplex(self._agent_factory)

    # -- POST ---------------------------------------------------------------

    async def handle_post(
        self,
        message: Any,
        *,
        content_type: str | None,
        connection_id: str | None,
        session_id: str | None,
    ) -> PostResult:
        if content_type is None or not content_type.lower().startswith("application/json"):
            return PostResult(415, {"error": "Content-Type must be application/json"})
        if isinstance(message, list):
            return PostResult(501, {"error": "Batch requests are not supported"})
        if not isinstance(message, dict):
            return PostResult(400, {"error": "Invalid JSON-RPC message"})

        if is_initialize_request(message):
            return await self._handle_initialize(message)

        if connection_id is None:
            return PostResult(400, {"error": "Missing connection id"})
        state = self._registry.get(connection_id)
        if state is None:
            return PostResult(404, {"error": "Unknown connection id"})

        method = message.get("method")
        if method_requires_session_header(method) and session_id is None:
            return PostResult(400, {"error": "Missing session id header"})
        if session_id is not None and not state.has_session(session_id):
            # A session-scoped POST references an unknown session.
            return PostResult(404, {"error": "Unknown session id"})

        await state.deliver_to_agent(message)
        return PostResult(202)

    async def _handle_initialize(self, message: dict[str, Any]) -> PostResult:
        state = self._registry.create(self._agent_factory)
        # Deliver initialize to the agent and await its response so we can return
        # the 200 body synchronously (initialize is the one blocking POST). If the
        # agent never responds (timeout) or errors, tear the just-created
        # connection down instead of leaking its pump task + agent connection.
        try:
            response = await state.request_response(message)
        except TimeoutError:
            await self._registry.remove(state.connection_id)
            return PostResult(504, {"error": "initialize timed out"})
        except Exception:
            await self._registry.remove(state.connection_id)
            return PostResult(500, {"error": "initialize failed"})
        return PostResult(200, response, {CONNECTION_ID_HEADER: state.connection_id})

    # -- GET / SSE ----------------------------------------------------------

    def validate_stream(self, *, connection_id: str | None, session_id: str | None) -> PostResult | None:
        """Validate a GET SSE request. Returns an error PostResult, or None if OK."""
        if connection_id is None:
            return PostResult(400, {"error": "Missing connection id"})
        state = self._registry.get(connection_id)
        if state is None:
            return PostResult(404, {"error": "Unknown connection id"})
        if session_id is not None and not state.has_session(session_id):
            return PostResult(404, {"error": "Unknown session id"})
        return None

    async def open_stream(
        self,
        *,
        connection_id: str,
        session_id: str | None,
    ) -> AsyncIterator[bytes]:
        """Yield SSE byte frames for a connection- or session-scoped stream.

        Emits a keepalive comment whenever the stream is idle for longer than
        :data:`SSE_KEEPALIVE_INTERVAL_SECONDS` so that idle-timeout intermediaries
        (proxies, load balancers) do not close an otherwise-healthy stream.
        """
        state = self._registry.get(connection_id)
        if state is None:
            return
        stream = state.ensure_session_stream(session_id) if session_id is not None else state.connection_stream
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

    async def handle_delete(self, *, connection_id: str | None) -> PostResult:
        if connection_id is None:
            return PostResult(400, {"error": "Missing connection id"})
        if self._registry.get(connection_id) is None:
            return PostResult(404, {"error": "Unknown connection id"})
        await self._registry.remove(connection_id)
        return PostResult(202)

    async def close(self) -> None:
        await self._registry.close_all()
