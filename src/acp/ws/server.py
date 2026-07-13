"""WebSocket server handling for the ASGI adapter (port of #155 ws-server.ts).

On upgrade we create a fresh :class:`~acp.http.server.ConnectionState` (bound to
its own ``AgentSideConnection``), accept the socket with an ``Acp-Connection-Id``
header, then pump JSON-RPC text frames both directions.  All server→client
traffic (across the connection- and every session-scoped stream) is multiplexed
onto the single socket.  On disconnect the connection and its sessions are torn
down.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..http.protocol import CONNECTION_ID_HEADER

if TYPE_CHECKING:
    from ..http.server import AcpServer, ConnectionState

__all__ = ["handle_asgi_websocket"]


async def handle_asgi_websocket(
    server: AcpServer,
    scope: dict[str, Any],
    receive: Callable,
    send: Callable,
) -> None:
    """Handle an ASGI ``websocket`` scope by bridging it to a new ACP connection."""
    # Wait for the connect message.
    message = await receive()
    if message["type"] != "websocket.connect":
        return

    state = server.create_websocket_connection()
    await send({
        "type": "websocket.accept",
        "headers": [(CONNECTION_ID_HEADER.lower().encode(), state.connection_id.encode())],
    })

    outbound_task = asyncio.ensure_future(_pump_outbound(state, send))
    try:
        await _pump_inbound(state, receive)
    finally:
        outbound_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await outbound_task
        await server.registry.remove(state.connection_id)


async def _pump_inbound(state: ConnectionState, receive: Callable) -> None:
    """Read client→server text frames and deliver them to the agent."""
    while True:
        message = await receive()
        msg_type = message["type"]
        if msg_type == "websocket.disconnect":
            return
        if msg_type != "websocket.receive":
            continue
        text = message.get("text")
        if text is None:
            # Ignore binary frames.
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            await state.deliver_to_agent(payload)


async def _pump_outbound(state: ConnectionState, send: Callable) -> None:
    """Forward all agent→client messages onto the socket as text frames."""
    async for message in state.iter_all_outbound():
        await send({"type": "websocket.send", "text": json.dumps(message, separators=(",", ":"))})
