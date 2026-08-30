# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#     "agent-client-protocol[http]",
#     "hypercorn>=0.17",
# ]
# ///
"""Serve an ACP agent over Streamable HTTP + WebSocket (experimental).

Run with an HTTP/2-capable ASGI server for spec-compliant Streamable HTTP.  This
example uses Hypercorn; Uvicorn works for WebSocket but does not serve HTTP/2.

    uv run examples/http_server.py
    # then, in another terminal:
    uv run examples/http_client.py
    uv run examples/ws_client.py
"""

import asyncio
from typing import Any
from uuid import uuid4

from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
)
from acp.http.asgi import create_asgi_app
from acp.interfaces import Client
from acp.schema import AgentMessageChunk, ClientCapabilities, Implementation, TextContentBlock


class EchoAgent(Agent):
    _conn: Client

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(self, cwd: str = "", **kwargs: Any) -> NewSessionResponse:
        return NewSessionResponse(session_id=uuid4().hex)

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        for block in prompt:
            text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
            await self._conn.session_update(
                session_id=session_id,
                update=AgentMessageChunk(content=TextContentBlock(text=f"echo: {text}")),
            )
        return PromptResponse(stop_reason="end_turn")


# One agent instance per connection.
app = create_asgi_app(lambda conn: EchoAgent())


async def main() -> None:
    import hypercorn.asyncio
    from hypercorn.config import Config

    config = Config()
    config.bind = ["localhost:8000"]
    # Enable HTTP/2 (Streamable HTTP requires it). Hypercorn negotiates h2c/h2.
    config.alpn_protocols = ["h2", "http/1.1"]
    print("Serving ACP agent on http://localhost:8000/acp (HTTP + WS)")
    await hypercorn.asyncio.serve(app, config)


if __name__ == "__main__":
    asyncio.run(main())
