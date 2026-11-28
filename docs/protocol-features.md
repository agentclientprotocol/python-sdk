# Protocol Features (v1)

The stable v1 bindings track [`schema-v1.35.0`](https://github.com/agentclientprotocol/agent-client-protocol/releases/tag/schema-v1.35.0). Handlers and connection methods accept expanded snake_case parameters, and extra keyword arguments serialize into JSON-RPC `_meta`.

## Unstable protocol methods

Core session, authentication, filesystem, and terminal methods work out of the box, including `session/delete` (`delete_session`) and `logout`. Unstable protocol methods—such as session `fork`/`archive`/`status`/`options`/`set_title`, `providers/*`, `auth/status`, elicitation, and `mcp/message`—are enabled on the receiving connection by passing `use_unstable_protocol=True`:

```python
await run_agent(MyAgent(), use_unstable_protocol=True)
```

Check the peer's advertised capabilities from `initialize` before calling optional or unstable methods. Unimplemented request handlers return JSON-RPC method-not-found (`-32601`).

## Session notices

When a client advertises `ClientCapabilities(session=ClientSessionCapabilities(notices=NoticeCapabilities()))`, agents can send live advisory notices outside session history:

```python
from acp.schema import SessionUpdateNotice

await client.session_update(
    session_id=session_id,
    update=SessionUpdateNotice(
        severity="warning",
        title="Context is nearly full",
        description="Start a new session soon.",
    ),
)
```

`title` must be non-empty; `severity` accepts `"info"`, `"warning"`, `"error"`, and custom string values.

## Stateless MCP-over-ACP

The unstable MCP binding targets **MCP 2026-07-28**. When an agent advertises `mcpCapabilities.acp`, a client can pass `AcpMcpServer(name=..., server_id=...)` during session setup after registering its `mcp_message` handler.

Agents send `mcp_message` requests with a fresh string `request_id` per operation and MCP metadata in `params["_meta"]`:

```python
from uuid import uuid4
from acp import McpErrorResponse

response = await conn.mcp_message(
    server_id="project-tools",
    request_id=uuid4().hex,
    method="tools/list",
    params={
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {},
        },
    },
)
if isinstance(response, McpErrorResponse):
    handle_mcp_error(response.error)
else:
    use_mcp_result(response.result)
```

Providers return `McpResultResponse(result=...)` or `McpErrorResponse(error=McpError(code=..., message=...))` and can send in-flight progress via `notify_mcp(...)` with matching `server_id` and `request_id`. Outer ACP failures raise `RequestError`; inner MCP errors stay inside `McpErrorResponse.error`.

Run `uv run python examples/mcp_over_acp.py` for a full local example.

## Request cancellation

Connections handle `$/cancel_request` automatically:

- Cancelling a task awaiting an outgoing request sends a best-effort `$/cancel_request` to the peer and raises `asyncio.CancelledError` locally.
- Receiving `$/cancel_request` cancels the handler task; if the handler catches `asyncio.CancelledError` and returns a value, that result is sent back, otherwise the peer receives `-32800` (`Request cancelled`).

Use `conn.cancel(session_id=...)` (`session/cancel`) to stop active work inside a prompt turn.

## Extension methods

Implement `ext_method` and `ext_notification` on your `Agent` or `Client` to handle vendor-prefixed `_` methods, and call `conn.ext_method(...)` / `conn.ext_notification(...)` to send them.
