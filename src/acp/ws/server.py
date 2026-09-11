"""Bind a Starlette WebSocket to an ACP agent connection."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from starlette.websockets import WebSocket, WebSocketState

from ..agent.connection import AgentSideConnection
from ..http.protocol import CONNECTION_ID_HEADER

if TYPE_CHECKING:
    from ..http.server import AgentFactory

__all__ = ["handle_websocket"]


class _WebSocketTransport:
    """Adapt Starlette's WebSocket to the message-level Transport interface."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    async def send(self, message: dict[str, Any]) -> None:
        if self._ws.client_state == WebSocketState.DISCONNECTED:
            raise ConnectionError("Transport closed")
        await self._ws.send_json(message)

    async def receive(self) -> dict[str, Any] | None:
        while self._ws.client_state != WebSocketState.DISCONNECTED:
            event = await self._ws.receive()
            if event["type"] == "websocket.disconnect":
                return None
            if event.get("text") is None:
                continue
            try:
                message = json.loads(event["text"])
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict):
                return message
        return None

    async def close(self) -> None:
        if self._ws.client_state == self._ws.application_state == WebSocketState.CONNECTED:
            await self._ws.close()


async def handle_websocket(agent_factory: AgentFactory, websocket: WebSocket) -> None:
    """Run one agent for the lifetime of the socket; disconnect cancels its work."""
    await websocket.accept(headers=[(CONNECTION_ID_HEADER.lower().encode(), uuid.uuid4().hex.encode())])
    transport = _WebSocketTransport(websocket)
    conn = AgentSideConnection(agent_factory, transport, listening=False)
    try:
        await conn.listen()
    finally:
        await conn.close()
