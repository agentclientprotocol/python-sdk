"""Streamable HTTP client transport (port of #155 ``http-stream.ts``, minus retry).

``create_http_stream(url, ...)`` returns a :class:`~acp._transport.Transport`
that can be handed to :func:`acp.connect_to_agent`.  The flow:

* The first message MUST be ``initialize``: POSTed as ``application/json``,
  expecting ``200 OK`` + an ``Acp-Connection-Id`` response header.  The JSON body
  is enqueued back into ``receive()`` so the core correlates it by ``id``.
* Subsequent messages are POSTed with the connection id (+ session id header for
  session-scoped methods) and return ``202 Accepted``.
* After ``initialize`` the client opens the connection-scoped SSE stream (GET);
  when it sees a new ``sessionId`` it opens that session-scoped SSE stream too.
* Server→client messages arrive on those SSE streams, merged into one
  ``receive()`` feed.  Order is preserved within a stream, interleaved across.
* A single SSE attempt is made per stream; on EOF/closure the reader surfaces
  end-of-stream (no auto-retry — that is the caller's responsibility).
* ``close()`` aborts in-flight streams and DELETEs the connection.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from .._sse import parse_sse_stream
from .protocol import (
    CONNECTION_ID_HEADER,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_SSE,
    SESSION_ID_HEADER,
    is_initialize_request,
    method_requires_session_header,
    session_id_from_message,
)

try:
    import httpx
except ImportError as exc:  # pragma: no cover - exercised via import guard message
    msg = "The Streamable HTTP transport requires the 'http' extra: pip install agent-client-protocol[http]"
    raise ImportError(msg) from exc

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .._transport import Transport

__all__ = ["AcpHttpStatusError", "create_http_stream"]

_EOF = object()


class AcpHttpStatusError(RuntimeError):
    """Raised when an HTTP request returns an unexpected status code."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


class _HttpStreamTransport:
    """Streamable HTTP client transport implementing the :class:`Transport` protocol."""

    def __init__(
        self,
        url: str,
        *,
        client: httpx.AsyncClient,
        owns_client: bool,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._url = url
        self._client = client
        self._owns_client = owns_client
        self._extra_headers = dict(headers or {})
        self._connection_id: str | None = None
        self._closed = False
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()
        self._stream_tasks: set[asyncio.Task[None]] = set()
        self._session_streams: set[str] = set()

    # -- Transport protocol -------------------------------------------------

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise ConnectionError("Transport closed")
        if is_initialize_request(message):
            await self._send_initialize(message)
            return
        await self._send_post(message)

    async def receive(self) -> dict[str, Any] | None:
        item = await self._inbox.get()
        if item is _EOF:
            return None
        return item

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in list(self._stream_tasks):
            task.cancel()
        for task in list(self._stream_tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._stream_tasks.clear()
        if self._connection_id is not None:
            with contextlib.suppress(Exception):
                await self._client.request(
                    "DELETE",
                    self._url,
                    headers={CONNECTION_ID_HEADER: self._connection_id, **self._extra_headers},
                )
        if self._owns_client:
            with contextlib.suppress(Exception):
                await self._client.aclose()
        self._inbox.put_nowait(_EOF)

    # -- Internals ----------------------------------------------------------

    async def _send_initialize(self, message: dict[str, Any]) -> None:
        headers = {"Content-Type": CONTENT_TYPE_JSON, **self._extra_headers}
        response = await self._client.post(self._url, json=message, headers=headers)
        if response.status_code != 200:
            raise AcpHttpStatusError(response.status_code, "initialize failed")
        connection_id = response.headers.get(CONNECTION_ID_HEADER)
        if not connection_id:
            raise AcpHttpStatusError(response.status_code, f"missing {CONNECTION_ID_HEADER} header")
        self._connection_id = connection_id
        body = response.json()
        # Enqueue the initialize result so the core correlates it by id.
        self._inbox.put_nowait(body)
        # Open the connection-scoped SSE stream.
        self._open_stream(session_id=None)

    async def _send_post(self, message: dict[str, Any]) -> None:
        if self._connection_id is None:
            raise ConnectionError("Cannot send before initialize established a connection id")
        headers = {"Content-Type": CONTENT_TYPE_JSON, CONNECTION_ID_HEADER: self._connection_id, **self._extra_headers}
        method = message.get("method")
        session_id = session_id_from_message(message)
        if method_requires_session_header(method) and session_id is not None:
            headers[SESSION_ID_HEADER] = session_id
        response = await self._client.post(self._url, json=message, headers=headers)
        if response.status_code not in (200, 202):
            raise AcpHttpStatusError(response.status_code, f"POST {method} failed")
        # Some servers may answer initialize-like 200 bodies; for 200 with a body enqueue it.
        if response.status_code == 200 and response.content:
            with contextlib.suppress(Exception):
                self._inbox.put_nowait(response.json())

    def _open_stream(self, *, session_id: str | None) -> None:
        if self._closed:
            return
        if session_id is not None:
            if session_id in self._session_streams:
                return
            self._session_streams.add(session_id)
        task = asyncio.ensure_future(self._consume_stream(session_id=session_id))
        self._stream_tasks.add(task)
        task.add_done_callback(self._stream_tasks.discard)

    async def _consume_stream(self, *, session_id: str | None) -> None:
        if self._connection_id is None:
            return
        headers = {
            "Accept": CONTENT_TYPE_SSE,
            CONNECTION_ID_HEADER: self._connection_id,
            **self._extra_headers,
        }
        if session_id is not None:
            headers[SESSION_ID_HEADER] = session_id
        try:
            async with self._client.stream("GET", self._url, headers=headers) as response:
                if response.status_code != 200:
                    return
                async for event in parse_sse_stream(_aiter_raw(response)):
                    self._handle_incoming(event)
        except (httpx.HTTPError, asyncio.CancelledError):
            return
        finally:
            self._on_stream_closed(session_id)

    def _on_stream_closed(self, session_id: str | None) -> None:
        """Handle a stream reader terminating (EOF, non-200, or error).

        A dropped/ended **connection-scoped** stream is the client's only channel
        for connection-level server→client messages, so its loss is surfaced as
        end-of-stream: ``receive()`` returns ``None``, the core's receive loop
        exits, and pending requests are rejected instead of hanging forever. v1
        does not auto-reconnect — that is the host's responsibility — but the
        disconnect must be observable. Session-scoped streams may legitimately
        close, so their EOF is not treated as a connection-level disconnect.
        """
        if session_id is not None:
            self._session_streams.discard(session_id)
            return
        if not self._closed:
            self._inbox.put_nowait(_EOF)

    def _handle_incoming(self, message: dict[str, Any]) -> None:
        # Open a session-scoped stream when any message carries a new sessionId
        # (e.g. a session/new or session/load result on the connection stream).
        session_id = session_id_from_message(message)
        if session_id is not None and session_id not in self._session_streams:
            self._open_stream(session_id=session_id)
        self._inbox.put_nowait(message)


async def _aiter_raw(response: httpx.Response) -> AsyncIterator[bytes]:
    async for chunk in response.aiter_bytes():
        yield chunk


def create_http_stream(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    headers: dict[str, str] | None = None,
) -> Transport:
    """Create a Streamable HTTP client :class:`Transport`.

    Args:
        url: The ACP endpoint URL (e.g. ``https://host/acp``).
        client: An optional pre-configured ``httpx.AsyncClient``.  If omitted, an
            HTTP/2-enabled client with a cookie jar is created and owned by the
            transport (closed on ``close()``).
        headers: Extra headers sent on every request.

    Returns:
        A :class:`Transport` usable with :func:`acp.connect_to_agent`.
    """
    owns_client = client is None
    if client is None:
        # SSE GET streams are long-lived, so disable read timeouts by default.
        client = httpx.AsyncClient(http2=True, timeout=httpx.Timeout(None))
    return _HttpStreamTransport(url, client=client, owns_client=owns_client, headers=headers)
