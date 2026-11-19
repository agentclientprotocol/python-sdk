"""Run a native MCP request and progress notification over an in-memory ACP connection.

Run with: uv run python examples/mcp_over_acp.py
No MCP initialization, HTTP listener, or external provider is needed.
"""

import asyncio
import logging
from typing import Any
from uuid import uuid4

from acp import Client, McpError, McpErrorResponse, McpResultResponse, MessageMcpResponse, RequestError
from acp._transport import memory_transport_pair
from acp.agent.connection import AgentSideConnection
from acp.client.connection import ClientSideConnection


class EchoProvider:
    connection: ClientSideConnection

    def on_connect(self, connection: ClientSideConnection) -> None:
        self.connection = connection

    async def mcp_message(
        self, server_id: str, request_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> MessageMcpResponse:
        if server_id != "echo":
            raise RequestError(-33001, "Server registration unavailable")
        params = params or {}
        meta = params.get("_meta")
        if not isinstance(meta, dict) or not isinstance(meta.get("io.modelcontextprotocol/protocolVersion"), str):
            return McpErrorResponse(error=McpError(code=-32602, message="Missing MCP protocol version"))
        if meta["io.modelcontextprotocol/protocolVersion"] != "2026-07-28":
            return McpErrorResponse(
                error=McpError(
                    code=-32022,
                    message="Unsupported MCP protocol version",
                    data={"supported": ["2026-07-28"], "requested": meta["io.modelcontextprotocol/protocolVersion"]},
                )
            )
        if not isinstance(meta.get("io.modelcontextprotocol/clientCapabilities"), dict):
            return McpErrorResponse(error=McpError(code=-32602, message="Missing MCP client capabilities"))
        if method != "tools/call" or params.get("name") != "echo":
            return McpErrorResponse(error=McpError(code=-32601, message="Unknown MCP operation"))
        await self.connection.notify_mcp(
            server_id=server_id,
            request_id=request_id,
            method="notifications/progress",
            params={"progressToken": meta["progressToken"], "progress": 1, "total": 1},
        )
        return McpResultResponse(
            result={
                "resultType": "complete",
                "content": [{"type": "text", "text": params["arguments"]["message"]}],
                "isError": False,
            }
        )


class Consumer:
    async def notify_mcp(
        self, server_id: str, request_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        logging.info("%s / %s: %s %s", server_id, request_id, method, params)


async def main() -> None:
    agent_transport, client_transport = memory_transport_pair()
    async with (
        AgentSideConnection(Consumer(), agent_transport, use_unstable_protocol=True) as agent,
        ClientSideConnection(EchoProvider(), client_transport, use_unstable_protocol=True),
    ):
        # The provider has already registered "echo" locally. In an application,
        # advertise AcpMcpServer(name="echo", server_id="echo") in session setup
        # after checking the agent's MCP-over-ACP capability.
        connection: Client = agent
        response = await connection.mcp_message(
            server_id="echo",
            request_id=uuid4().hex,
            method="tools/call",
            params={
                "name": "echo",
                "arguments": {"message": "Hello over ACP"},
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                    "io.modelcontextprotocol/clientCapabilities": {},
                    "progressToken": "echo-progress",
                },
            },
        )
        if isinstance(response, McpErrorResponse):
            logging.info("MCP error: %s", response.error)
        else:
            logging.info("MCP result: %s", response.result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
