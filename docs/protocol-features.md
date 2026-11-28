# Protocol Features (v1)

The stable v1 bindings track [`schema-v1.35.0`](https://github.com/agentclientprotocol/agent-client-protocol/releases/tag/schema-v1.35.0). Both `Agent` and `Client` interfaces expose expanded snake_case parameters, with extra keyword arguments carrying JSON-RPC `_meta`.

## Enabling unstable protocol methods

Unstable methods require explicit opt-in on the receiving side by passing `use_unstable_protocol=True` to `run_agent()`, `connect_to_agent()`, `AgentSideConnection`, or `ClientSideConnection`. Check the peer's advertised capabilities from `initialize` before calling optional or unstable methods. Unimplemented request handlers return JSON-RPC method-not-found (`-32601`).

### Agent methods

| Wire method | Python method | Stability |
| --- | --- | --- |
| `initialize` | `initialize` | Stable |
| `authenticate`, `logout` | `authenticate`, `logout` | Stable |
| `session/new`, `session/load`, `session/list` | `new_session`, `load_session`, `list_sessions` | Stable |
| `session/resume`, `session/close`, `session/delete` | `resume_session`, `close_session`, `delete_session` | Stable |
| `session/set_mode`, `session/set_config_option` | `set_session_mode`, `set_config_option` | Stable |
| `session/prompt`, `session/cancel` | `prompt`, `cancel` | Stable |
| `session/fork` | `fork_session` | Unstable |
| `session/archive`, `session/unarchive` | `archive_session`, `unarchive_session` | Unstable |
| `session/status`, `session/options`, `session/set_title` | `session_status`, `session_options`, `set_session_title` | Unstable |
| `auth/status` | `auth_status` | Unstable |
| `providers/list`, `providers/set`, `providers/disable` | `list_providers`, `set_provider`, `disable_provider` | Unstable |
| `mcp/message` (notification) | `notify_mcp` | Unstable |

### Client methods

| Wire method | Python method | Stability |
| --- | --- | --- |
| `session/request_permission`, `session/update` | `request_permission`, `session_update` | Stable |
| `fs/read_text_file`, `fs/write_text_file` | `read_text_file`, `write_text_file` | Stable |
| `terminal/create`, `terminal/output`, `terminal/wait_for_exit`, `terminal/kill`, `terminal/release` | `create_terminal`, `terminal_output`, `wait_for_terminal_exit`, `kill_terminal`, `release_terminal` | Stable |
| `elicitation/create`, `elicitation/complete` | `create_elicitation`, `complete_elicitation` | Unstable |
| `auth/status_update` (notification) | `auth_status_update` | Unstable |
| `mcp/message` (request) | `mcp_message` | Unstable |

For example, an agent supporting session deletion implements `delete_session`:

```python
from typing import Any
from acp import DeleteSessionResponse


async def delete_session(self, session_id: str, **kwargs: Any) -> DeleteSessionResponse:
    await self.session_store.delete(session_id)
    return DeleteSessionResponse()
```

Clients invoke `await conn.delete_session(session_id=session_id)` to remove persisted session state, and `await conn.close_session(session_id=session_id)` to close an active session handle.

## Session notices

Clients that render live advisory notices advertise `ClientCapabilities(session=ClientSessionCapabilities(notices=NoticeCapabilities()))` during `initialize`. When that capability is advertised, agents can send `SessionUpdateNotice` updates:

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

Notices are live advisory events outside retained session history. `title` must be non-empty; `severity` accepts `"info"`, `"warning"`, `"error"`, and custom string values.

## Stateless MCP-over-ACP

The unstable native MCP binding targets **MCP 2026-07-28**. A client/provider declares an `AcpMcpServer(name=..., server_id=...)` during session setup when the agent advertises `mcpCapabilities.acp`, after registering its `mcp_message` handler so immediate calls from the agent are served.

Agents send `mcp_message` requests with a fresh string `request_id` per operation and the required MCP metadata in `params["_meta"]`:

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
    trace="outer ACP metadata",
)
if isinstance(response, McpErrorResponse):
    handle_mcp_error(response.error)
else:
    use_mcp_result(response.result)
```

Providers answer with `McpResultResponse(result=...)` or `McpErrorResponse(error=McpError(code=..., message=...))`. Outer ACP transport or routing errors raise `RequestError`; inner MCP errors stay inside `McpErrorResponse.error`. While a request is active, providers can emit progress via `notify_mcp(server_id=..., request_id=..., method=..., params=...)` using matching IDs, stopping notifications before returning the final response.

MCP-over-ACP works across stdio, Streamable HTTP, and WebSocket connections. Run `uv run python examples/mcp_over_acp.py` for a runnable demo, and see the [ACP MCP-over-ACP specification](https://agentclientprotocol.com/rfds/mcp-over-acp) for registration and error-code rules.

## Request cancellation

Connections handle the protocol's `$/cancel_request` notification automatically:

- **Incoming cancellation**: The SDK cancels the asyncio task running the target handler. If the handler catches `asyncio.CancelledError` and returns a value, that result is sent to the peer; otherwise the peer receives `-32800` (`Request cancelled`).
- **Outgoing cancellation**: Cancelling the task awaiting an outgoing request (for example via `asyncio.wait_for`) raises `asyncio.CancelledError` locally and sends a best-effort `$/cancel_request` notification to the peer.

Use `session/cancel` (`conn.cancel(session_id=...)`) to stop active work inside a session turn.

## Extension methods

Custom vendor methods use the `_` prefix required by ACP. Implement `ext_method(self, method: str, params: dict[str, Any])` and `ext_notification(self, method: str, params: dict[str, Any])` on your `Agent` or `Client` to handle incoming extension traffic, and call `conn.ext_method(...)` / `conn.ext_notification(...)` to send outgoing extensions.
