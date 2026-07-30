"""Thin ASGI adapter bridging Starlette/FastAPI/Hypercorn to :class:`AcpServer`.

``create_asgi_app(agent_factory)`` returns an ASGI 3.0 application callable that
handles POST/GET/DELETE (and WebSocket upgrades) on the ACP endpoint.  Users can
mount it directly or wrap it in their framework of choice.

Note: for a spec-compliant Streamable HTTP server, run this under an
HTTP/2-capable ASGI server (Hypercorn, Daphne, Granian) or terminate HTTP/2 at a
proxy.  Uvicorn does not serve HTTP/2 (WebSocket still works).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .protocol import CONNECTION_ID_HEADER, CONTENT_TYPE_SSE, SESSION_ID_HEADER
from .server import AcpServer

if TYPE_CHECKING:
    from .server import AgentFactory

__all__ = ["AcpAsgiApp", "create_asgi_app"]

_JSON_HEADERS = [(b"content-type", b"application/json")]


def _header_lookup(scope_headers: list[tuple[bytes, bytes]], name: str) -> str | None:
    target = name.lower().encode()
    for key, value in scope_headers:
        if key.lower() == target:
            return value.decode("latin-1")
    return None


class AcpAsgiApp:
    """ASGI application wrapping an :class:`AcpServer`."""

    def __init__(self, server: AcpServer) -> None:
        self._server = server

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        scope_type = scope["type"]
        if scope_type == "lifespan":
            await self._handle_lifespan(receive, send)
            return
        if scope_type == "websocket":
            await self._handle_websocket(scope, receive, send)
            return
        if scope_type != "http":
            return

        method = scope["method"]
        if method == "POST":
            await self._handle_post(scope, receive, send)
        elif method == "GET":
            await self._handle_get(scope, receive, send)
        elif method == "DELETE":
            await self._handle_delete(scope, send)
        else:
            await self._send_json(send, 405, {"error": "Method not allowed"})

    async def _handle_lifespan(self, receive: Callable, send: Callable) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self._server.close()
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _read_body(self, receive: Callable) -> bytes:
        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break
        return b"".join(chunks)

    async def _handle_post(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        headers = scope["headers"]
        content_type = _header_lookup(headers, "content-type")
        connection_id = _header_lookup(headers, CONNECTION_ID_HEADER)
        session_id = _header_lookup(headers, SESSION_ID_HEADER)
        raw = await self._read_body(receive)
        try:
            message = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            await self._send_json(send, 400, {"error": "Invalid JSON"})
            return
        result = await self._server.handle_post(
            message,
            content_type=content_type,
            connection_id=connection_id,
            session_id=session_id,
        )
        await self._send_json(send, result.status, result.body, extra_headers=result.headers)

    async def _handle_get(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        headers = scope["headers"]
        upgrade = _header_lookup(headers, "upgrade")
        if upgrade is not None and upgrade.lower() == "websocket":
            # WebSocket upgrades arrive as scope type "websocket" in ASGI; a GET
            # http scope with Upgrade is non-standard, so reject clearly.
            await self._send_json(send, 400, {"error": "WebSocket upgrade must use the ws scope"})
            return
        accept = _header_lookup(headers, "accept") or ""
        if CONTENT_TYPE_SSE not in accept and "*/*" not in accept:
            await self._send_json(send, 406, {"error": "Accept must include text/event-stream"})
            return
        connection_id = _header_lookup(headers, CONNECTION_ID_HEADER)
        session_id = _header_lookup(headers, SESSION_ID_HEADER)
        error = self._server.validate_stream(connection_id=connection_id, session_id=session_id)
        if error is not None:
            await self._send_json(send, error.status, error.body)
            return
        if connection_id is None:  # validated above, narrow for type-checker
            await self._send_json(send, 400, {"error": "Missing connection id"})
            return
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", CONTENT_TYPE_SSE.encode()),
                (b"cache-control", b"no-cache"),
                (b"connection", b"keep-alive"),
            ],
        })
        async for frame in self._server.open_stream(connection_id=connection_id, session_id=session_id):
            await send({"type": "http.response.body", "body": frame, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _handle_delete(self, scope: dict[str, Any], send: Callable) -> None:
        connection_id = _header_lookup(scope["headers"], CONNECTION_ID_HEADER)
        result = await self._server.handle_delete(connection_id=connection_id)
        await self._send_json(send, result.status, result.body, extra_headers=result.headers)

    async def _handle_websocket(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        from ..ws.server import handle_asgi_websocket

        await handle_asgi_websocket(self._server, scope, receive, send)

    async def _send_json(
        self,
        send: Callable,
        status: int,
        body: dict[str, Any] | None,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        payload = json.dumps(body).encode() if body is not None else b""
        headers = list(_JSON_HEADERS)
        if extra_headers:
            headers.extend((k.encode("latin-1"), v.encode("latin-1")) for k, v in extra_headers.items())
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": payload})


def create_asgi_app(agent_factory: AgentFactory) -> AcpAsgiApp:
    """Create an ASGI app serving an ACP agent over Streamable HTTP + WebSocket.

    Args:
        agent_factory: Called once per connection with the bound
            ``AgentSideConnection`` to produce a per-connection ``Agent``.

    Returns:
        An :class:`AcpAsgiApp` ASGI 3.0 application.
    """
    return AcpAsgiApp(AcpServer(agent_factory))
