"""WebSocket client cookie support (RFD §5: cookies MUST work on WS transport)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from websockets.asyncio.server import serve

from acp._cookies import MemoryAcpCookieStore
from acp.ws.client import create_websocket_stream


@pytest.mark.asyncio
async def test_ws_client_captures_set_cookie_from_handshake() -> None:
    """A caller-owned cookie store must capture Set-Cookie from the upgrade response."""

    def process_response(connection: Any, request: Any, response: Any) -> Any:
        response.headers["Set-Cookie"] = "affinity=abc123; Path=/"
        return response

    async def handler(ws: Any) -> None:
        await ws.close()

    store = MemoryAcpCookieStore()
    async with serve(handler, "localhost", 0, process_response=process_response) as server:
        port = server.sockets[0].getsockname()[1]
        transport = await create_websocket_stream(f"ws://localhost:{port}", cookie_store=store)
        try:
            assert store.cookie_header() == "affinity=abc123"
        finally:
            await transport.close()


@pytest.mark.asyncio
async def test_ws_client_sends_stored_cookie_on_handshake() -> None:
    """Stored cookies must be echoed back as a Cookie header on the next handshake."""
    seen: dict[str, Any] = {}

    async def handler(ws: Any) -> None:
        seen["cookie"] = ws.request.headers.get("Cookie")
        await ws.close()

    store = MemoryAcpCookieStore()
    store.store_set_cookie("affinity=abc123")

    async with serve(handler, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = await create_websocket_stream(f"ws://localhost:{port}", cookie_store=store)
        try:
            await asyncio.sleep(0.05)
            assert seen.get("cookie") == "affinity=abc123"
        finally:
            await transport.close()
