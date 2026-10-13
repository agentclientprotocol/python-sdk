# Transport architecture

Stdio uses byte streams between a client and an agent process. The experimental
HTTP and WebSocket adapters expose message transports to the same ACP connection
layer. See the [web transport guide](../reference/experimental/web-transports.md) for setup and APIs.

## HTTP and WebSocket

Start reading at `acp/http/asgi.py`. It creates Starlette routes, passes parsed
HTTP requests to `AcpServer`, and binds WebSockets in `acp/ws/server.py`. Both use
`AgentSideConnection` and its message-level `Transport` interface:

```text
HTTP POST → _HttpTransport incoming queue → AgentSideConnection → agent
HTTP GET  ← StreamingResponse ← SSE buffer ← _HttpTransport.send() ← agent output

Starlette WebSocket ↔ _WebSocketTransport ↔ AgentSideConnection ↔ agent
```

For HTTP, `AcpServer` owns a dictionary of active connections. Each connection
has one incoming queue and one SSE buffer per stream. The incoming queue lets
POST return `202` while the agent handles the request. Output goes directly to
the relevant SSE buffer.

HTTP output follows these routing rules:

| Message | Destination | Why |
| --- | --- | --- |
| `initialize` response | POST body, via one Future | Establishes the connection before GET streams open |
| Response containing a new `sessionId` | Connection SSE stream | The client needs the ID before it can open the session stream |
| `session/load` replay and response, `session/resume` response | Connection SSE stream | Replay precedes the response; the client gets the session ID from the original request |
| Other messages | Session SSE stream when known, otherwise connection stream | Responses use their request's recorded session; requests/notifications carry `sessionId` |

`OutboundStream` retains a bounded buffer, backpressure, and close handling.
Idle SSE streams emit keepalives. These support slow readers, streams that open
after messages arrive, and orderly teardown. `DELETE` and server shutdown close
the HTTP connections and cancel their agent work.

The transport retains connection and stream state, not conversation history.
Reopening a transport creates a new connection; loading or resuming a session
depends on the agent's storage and advertised capabilities. A dropped
connection-scoped SSE stream ends the client's receive loop and rejects pending
requests. A session stream ending does not close the whole connection. The SDK
does not retry either stream automatically.

WebSocket already provides one bidirectional stream. Its transport adapts
Starlette's socket to JSON-RPC messages; it needs no HTTP connection registry,
session routing, SSE buffers, or multiplex mode. The ASGI handler owns the agent
connection and closes it on socket disconnect or handler cancellation.
