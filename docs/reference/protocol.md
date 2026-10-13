# Protocol features and cancellation

The default `acp` runtime uses protocol v1. This page documents the bundled
[`schema-v1.23.0`](https://github.com/agentclientprotocol/agent-client-protocol/blob/schema-v1.23.0/schema/v1/schema.unstable.json) bindings on `main`. The official [v1 documentation](https://agentclientprotocol.com/protocol/v1/overview) describes the protocol. Check the peer's capabilities before using
optional methods; schema support alone does not mean a peer implements them.
For the separate draft runtime, see [experimental v2](experimental/protocol-v2.md).

## Session notices (unstable)

The v1 bindings use `schema-v1.23.0`. Clients that can display live advisory
notices advertise `ClientCapabilities(session=ClientSessionCapabilities(notices=NoticeCapabilities()))`
in `initialize`; these models are available from `acp.schema`.

After checking that capability, agents can send a notice:

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

Notices are live events, outside session history. The title must be non-empty;
severity accepts `info`, `warning`, `error`, and custom or future strings.
Custom severity values begin with `_`; other unknown values are reserved for
future protocol variants. Agents send notices only to clients that advertise
support, and cannot rely on them being displayed. Severity is a presentation
hint: an `error` notice does not fail a request or end a turn. See the
[session-notices contract](https://agentclientprotocol.com/rfds/session-notices).

Tool calls include the stable `name` field. Terminal authentication methods use
`TerminalAuthMethod`. Accepted elicitation content validates scalar values and
string lists; nested objects are not valid form values.

## Additional v1 protocol methods

The SDK exposes all agent and client methods in the bundled `schema-v1.23.0`
method tables. `session/delete` and `logout` are stable and do not require
`use_unstable_protocol=True`. The other methods in the table below are unstable;
enable that flag on the receiving connection (or `run_agent`) to route them.
Check the peer's advertised capabilities before calling these methods.

| Wire method | Python method | Receiver |
| --- | --- | --- |
| `session/delete` | `delete_session(session_id=...)` | Agent |
| `providers/list`, `providers/set`, `providers/disable` | `list_providers`, `set_provider`, `disable_provider` | Agent |
| `logout` | `logout()` | Agent |
| `mcp/connect`, `mcp/disconnect` | `connect_mcp`, `disconnect_mcp` | Client |
| `mcp/message` (request) | `mcp_message` | Either peer |
| `mcp/message` (notification) | `notify_mcp` | Either peer |
| `nes/start`, `nes/suggest`, `nes/close` | `start_nes`, `suggest_nes`, `close_nes` | Agent |
| `nes/accept`, `nes/reject` | `accept_nes`, `reject_nes` | Agent |
| `document/didOpen`, `document/didChange`, `document/didClose`, `document/didSave`, `document/didFocus` | `did_open`, `did_change`, `did_close`, `did_save`, `did_focus` | Agent |

For example, an agent that advertises session deletion implements:

```python
from acp import DeleteSessionResponse

async def delete_session(self, session_id: str, **kwargs) -> DeleteSessionResponse:
    await self.session_store.delete(session_id)
    return DeleteSessionResponse()
```

A client then calls `await connection.delete_session(session_id=session_id)`.
Deletion removes stored session data; `close_session` only closes the active
session. The SDK dispatches these calls to your implementation; it does not
provide session storage or provider management itself. Implement each method
you advertise; the interface declaration alone does not supply its behavior.
For the session lifecycle contracts, see [session setup](https://agentclientprotocol.com/protocol/v1/session-setup)
and [session deletion](https://agentclientprotocol.com/protocol/v1/session-delete).

MCP requests return the inner JSON result unchanged, including `null`. Use
`notify_mcp` for one-way MCP messages. Both APIs accept `connection_id`, `method`,
and optional `params`. These methods share the same connections and routers
across stdio, HTTP, and WebSocket transports.

## Request cancellation

Connections handle the protocol's `$/cancel_request` notification. When the
peer cancels one of its requests, the SDK cancels the task running your handler.
The handler may catch `asyncio.CancelledError` and return a (partial) result;
otherwise the peer receives a `-32800` "Request cancelled" error. Cancellations
for unknown or already finished requests are ignored; a cancelled request still
gets exactly one response unless the connection is closed first.

Cancelling the task that awaits an outgoing request raises `CancelledError`
locally without waiting for the peer, and additionally sends a best-effort
`$/cancel_request` for that request. When `asyncio.wait_for` times out, it cancels
the awaiting task and raises `TimeoutError` to its caller.
Cancellation support is optional for peers. See the protocol's
[request-cancellation contract](https://agentclientprotocol.com/protocol/v1/cancellation).
To stop a prompt turn, call `await connection.cancel(session_id=session_id)`.
This sends `session/cancel` to your agent's handler; the agent stops its work
and completes the original prompt with `stop_reason="cancelled"`. The client
keeps handling updates until that response and returns cancellation outcomes
for pending permission requests. See [turn cancellation](../explanation/lifecycle.md#cancel-a-turn).
