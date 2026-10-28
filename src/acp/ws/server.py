"""Bind a Starlette WebSocket to an ACP agent connection."""

from __future__ import annotations

import contextlib
import logging
import uuid
from typing import TYPE_CHECKING, Any

from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from .._jsonrpc import decode_message, error_response
from ..agent.connection import AgentSideConnection
from ..exceptions import RequestError
from ..http.protocol import CONNECTION_ID_HEADER

if TYPE_CHECKING:
    from ..experimental.negotiation import AgentProtocolRouter
    from ..http.server import AgentFactory

__all__ = ["handle_websocket"]


class _WebSocketTransport:
    """Adapt Starlette's WebSocket to the message-level Transport interface."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._warned_binary = False

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
                if event.get("bytes") is not None and not self._warned_binary:
                    self._warned_binary = True
                    logging.warning("Ignoring binary WebSocket frame; ACP messages must be sent as text frames")
                continue
            try:
                message = decode_message(event["text"])
                if message is not None:
                    return message
            except RequestError as error:
                # A failed write must not discard valid messages buffered by ASGI.
                if self._ws.application_state == WebSocketState.CONNECTED:
                    with contextlib.suppress(WebSocketDisconnect):
                        await self.send(error_response(error))
        return None

    async def close(self) -> None:
        if self._ws.client_state == self._ws.application_state == WebSocketState.CONNECTED:
            await self._ws.close()


async def handle_websocket(agent_factory: AgentFactory | AgentProtocolRouter, websocket: WebSocket) -> None:
    """Run one agent for the lifetime of the socket; disconnect cancels its work."""
    await websocket.accept(headers=[(CONNECTION_ID_HEADER.lower().encode(), uuid.uuid4().hex.encode())])
    transport = _WebSocketTransport(websocket)
    from ..experimental.negotiation import AgentProtocolRouter

    if isinstance(agent_factory, AgentProtocolRouter):
        conn = agent_factory.connect(transport, listening=False)
    else:
        conn = AgentSideConnection(agent_factory, transport, listening=False)
    try:
        await conn.listen()
    finally:
        await conn.close()
