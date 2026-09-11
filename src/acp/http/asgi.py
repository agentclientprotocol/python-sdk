"""Starlette application for ACP over Streamable HTTP and WebSocket.

Run the application directly or mount it under another ASGI application.
Use an HTTP/2-capable ASGI server for Streamable HTTP.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from ..ws.server import handle_websocket
from .protocol import ACP_ENDPOINT_PATH, CONNECTION_ID_HEADER, CONTENT_TYPE_SSE, SESSION_ID_HEADER
from .server import AcpServer, AgentFactory

__all__ = ["create_asgi_app"]


def create_asgi_app(agent_factory: AgentFactory, *, path: str = ACP_ENDPOINT_PATH) -> Starlette:
    """Create a Starlette app with one agent instance per connection.

    The app handles POST/GET/DELETE and WebSocket at ``path`` (default: /acp).
    The path is relative to any parent mount. When mounting the app, enter its
    lifespan from the parent lifespan too.
    """
    server = AcpServer(agent_factory)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await server.close()

    async def websocket(websocket: WebSocket) -> None:
        await handle_websocket(agent_factory, websocket)

    return Starlette(
        routes=[
            Route(path, partial(_post, server), methods=["POST"]),
            Route(path, partial(_get, server), methods=["GET"]),
            Route(path, partial(_delete, server), methods=["DELETE"]),
            WebSocketRoute(path, websocket),
        ],
        lifespan=lifespan,
    )


async def _post(server: AcpServer, request: Request) -> Response:
    try:
        message = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    return await server.handle_post(
        message,
        content_type=request.headers.get("content-type"),
        connection_id=request.headers.get(CONNECTION_ID_HEADER),
        session_id=request.headers.get(SESSION_ID_HEADER),
    )


async def _get(server: AcpServer, request: Request) -> Response:
    if request.headers.get("upgrade", "").lower() == "websocket":
        return JSONResponse({"error": "WebSocket upgrade must use the ws scope"}, status_code=400)
    accept = request.headers.get("accept", "")
    if CONTENT_TYPE_SSE not in accept and "*/*" not in accept:
        return JSONResponse({"error": "Accept must include text/event-stream"}, status_code=406)
    connection_id = request.headers.get(CONNECTION_ID_HEADER)
    if connection_id is None:
        return JSONResponse({"error": "Missing connection id"}, status_code=400)
    session_id = request.headers.get(SESSION_ID_HEADER)
    error = server.validate_stream(connection_id=connection_id, session_id=session_id)
    if error is not None:
        return error
    return StreamingResponse(
        server.open_stream(connection_id=connection_id, session_id=session_id),
        media_type=CONTENT_TYPE_SSE,
        headers={"Cache-Control": "no-cache"},
    )


async def _delete(server: AcpServer, request: Request) -> Response:
    return await server.handle_delete(connection_id=request.headers.get(CONNECTION_ID_HEADER))
