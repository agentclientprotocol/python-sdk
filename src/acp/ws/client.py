"""WebSocket client transport (port of #155 ``ws-stream.ts``).

``create_websocket_stream(url, ...)`` connects a WebSocket and returns a
:class:`~acp._transport.Transport`.  Messages are JSON-RPC **text** frames;
binary frames are ignored.  The client must still send ``initialize`` as the
first message over the socket.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any

from .._cookies import MemoryAcpCookieStore

try:
    import websockets
    from websockets.asyncio.client import connect as ws_connect
except ImportError as exc:  # pragma: no cover - exercised via import guard message
    msg = "The WebSocket transport requires the 'http' extra: pip install agent-client-protocol[http]"
    raise ImportError(msg) from exc

if TYPE_CHECKING:
    from .._transport import Transport

__all__ = ["MemoryAcpCookieStore", "create_websocket_stream"]

# Case-insensitive header name a server uses to set connection-affinity cookies
# on the WebSocket upgrade response.
_SET_COOKIE_HEADER = "Set-Cookie"


class _WebSocketTransport:
    """WebSocket client transport implementing the :class:`Transport` protocol."""

    def __init__(self, connection: Any) -> None:
        self._ws = connection
        self._closed = False

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise ConnectionError("Transport closed")
        await self._ws.send(json.dumps(message, separators=(",", ":")))

    async def receive(self) -> dict[str, Any] | None:
        while True:
            try:
                frame = await self._ws.recv()
            except websockets.ConnectionClosed:
                return None
            # Ignore binary frames; only text JSON-RPC is meaningful.
            if isinstance(frame, bytes):
                continue
            try:
                return json.loads(frame)
            except json.JSONDecodeError:
                continue

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._ws.close()


async def create_websocket_stream(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    cookie_store: MemoryAcpCookieStore | None = None,
) -> Transport:
    """Connect a WebSocket and return a :class:`Transport`.

    Per the RFD, clients MUST accept, store, and return cookies on all HTTP-based
    transports (including WebSocket) so servers can rely on cookies for session
    affinity (e.g. sticky sessions behind a load balancer). A single WebSocket is
    one long-lived connection, so cookie support matters across *reconnects*:
    pass a caller-owned ``cookie_store`` reused between fresh streams. Any cookies
    already in the store are sent as a ``Cookie`` header on the handshake, and any
    ``Set-Cookie`` headers on the upgrade response are captured back into it.

    Args:
        url: The ACP WebSocket endpoint (e.g. ``ws://host/acp``).
        headers: Extra headers sent during the handshake.
        cookie_store: Optional caller-owned affinity cookie store to reuse across
            reconnects. If omitted, an ephemeral per-stream store is used.

    Returns:
        A connected :class:`Transport` usable with :func:`acp.connect_to_agent`.
    """
    store = cookie_store if cookie_store is not None else MemoryAcpCookieStore()
    request_headers = dict(headers or {})
    cookie_header = store.cookie_header()
    if cookie_header and not _has_header(request_headers, "Cookie"):
        request_headers["Cookie"] = cookie_header
    connection = await ws_connect(url, additional_headers=request_headers or None)
    _capture_set_cookies(connection, store)
    return _WebSocketTransport(connection)


def _has_header(headers: dict[str, str], name: str) -> bool:
    lowered = name.lower()
    return any(key.lower() == lowered for key in headers)


def _capture_set_cookies(connection: Any, store: MemoryAcpCookieStore) -> None:
    """Store ``Set-Cookie`` headers from the WebSocket upgrade response."""
    response = getattr(connection, "response", None)
    response_headers = getattr(response, "headers", None)
    if response_headers is None:
        return
    get_all = getattr(response_headers, "get_all", None)
    values = list(get_all(_SET_COOKIE_HEADER)) if get_all is not None else []
    if values:
        store.store_set_cookies(values)
