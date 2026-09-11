# Web Transport (Streamable HTTP & WebSocket)

> **Experimental.** The remote web transports are experimental and may change.
> They ship as an optional extra and are import-guarded.

The SDK can run ACP over two remote connectivity profiles in addition to stdio:

- **Streamable HTTP** — `POST` for client→server messages, long-lived `GET` SSE
  streams for server→client messages (one connection-scoped stream plus one per
  session), and `DELETE` to terminate. `initialize` returns `200 OK` with a JSON
  body; all other POSTs return `202 Accepted`. **Requires HTTP/2.**
- **WebSocket** — a `GET` upgrade on the same endpoint carrying full-duplex
  JSON-RPC text frames.

Both reuse the existing JSON-RPC message format and ACP lifecycle
(`initialize` → session methods → close).

## Installation

```bash
pip install "agent-client-protocol[http]"
```

This pulls in `httpx[http2]` (HTTP/2 + SSE consumption), `websockets`, and
`starlette` (the server application). The core SDK and stdio transport do not
require these optional dependencies.

## Client

Both transports produce a message-level `Transport` that plugs into the existing
`connect_to_agent`:

```python
from acp import connect_to_agent
from acp.http import create_http_stream
from acp.ws import create_websocket_stream

# Streamable HTTP
transport = create_http_stream("http://localhost:8000/acp")
conn = connect_to_agent(my_client, transport)

# ...or WebSocket
transport = await create_websocket_stream("ws://localhost:8000/acp")
conn = connect_to_agent(my_client, transport)

init = await conn.initialize(protocol_version=1)
session = await conn.new_session(cwd="/tmp", mcp_servers=[])
await conn.prompt(session_id=session.session_id, prompt=[...])
await conn.close()
await transport.close()
```

The client sends `initialize` first, reads the `Acp-Connection-Id` response
header, then opens the connection-scoped SSE stream. When a new `sessionId`
appears it opens that session-scoped stream too. A single SSE attempt is made per
stream; reconnect/retry is the caller's responsibility (v1 of the RFD).

## Server

The server uses Starlette for HTTP requests, responses, routing, streaming,
WebSocket handling, and application lifespan:

```python
from acp.http.asgi import create_asgi_app

# One agent instance is created per connection.
app = create_asgi_app(lambda conn: MyAgent())
```

`app` is a `starlette.applications.Starlette` instance handling
`POST`/`GET`/`DELETE` and WebSocket upgrades at `/acp` by default. Set the
keyword-only `path` argument to use a different endpoint for both transports:

```python
app = create_asgi_app(lambda conn: MyAgent(), path="/rpc")
```

Other paths do not serve ACP. Starlette supplies
`Request`, `JSONResponse`, `StreamingResponse`, and `WebSocket`; the SDK keeps
ACP connection and session routing.

### Mounting in another application

Mount the app at the desired prefix. The parent must enter the child lifespan
so HTTP connections are cleaned up during shutdown (mounted application
lifespans are not run automatically):

```python
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.routing import Mount

acp_app = create_asgi_app(lambda conn: MyAgent())

@asynccontextmanager
async def lifespan(app):
    async with acp_app.router.lifespan_context(acp_app):
        yield

app = Starlette(routes=[Mount("/agents", app=acp_app)], lifespan=lifespan)
# Connect to /agents/acp using either HTTP or WebSocket.
```

With `path="/rpc"`, the mounted endpoint is `/agents/rpc`. Use `path="/"` to
serve ACP at the mount root (`/agents/`).

### How the server fits together

Start reading at `acp/http/asgi.py`. It creates Starlette routes, passes parsed
HTTP requests to `AcpServer`, and binds WebSockets in `acp/ws/server.py`. Both use the existing
`AgentSideConnection` and its message-level `Transport` interface:

```text
HTTP POST → _HttpTransport incoming queue → AgentSideConnection → agent
HTTP GET  ← StreamingResponse ← SSE buffer ← _HttpTransport.send() ← agent output

Starlette WebSocket ↔ _WebSocketTransport ↔ AgentSideConnection ↔ agent
```

For HTTP, `AcpServer` owns a dictionary of active connections. Each connection
has one incoming queue and one SSE buffer per stream. The incoming queue lets
POST return `202` while the agent handles the request. Output goes directly to
the relevant SSE buffer; there is no intermediate transport pair or pump task.

HTTP output needs three routing rules:

| Message | Destination | Why |
| --- | --- | --- |
| `initialize` response | POST body, via one Future | Establishes the connection before GET streams open |
| Response containing a new `sessionId` | Connection SSE stream | The client needs the ID before it can open the session stream |
| Other messages | Session SSE stream when known, otherwise connection stream | Responses use their request's recorded session; requests/notifications carry `sessionId` |

`OutboundStream` retains a bounded buffer, backpressure, and close handling.
Idle SSE streams emit keepalives. These support slow readers, streams that open
after messages arrive, and orderly teardown. `DELETE` and server shutdown close
the HTTP connections and cancel their agent work.

WebSocket already provides one bidirectional stream. Its transport adapts
Starlette's socket to JSON-RPC messages; it needs no HTTP connection registry,
session routing, SSE buffers, or multiplex mode. The ASGI handler owns the agent
connection and closes it on socket disconnect or handler cancellation.

### Simplification experiment

The original 80 HTTP, WebSocket, and RPC tests passed after each ablation.
The Starlette migration also passes these behaviors; assertions now inspect
Starlette response objects and WebSocket tests use the framework's socket:

| Stage | Removed | Lines across the three server files |
| --- | --- | ---: |
| Baseline | — | 715 |
| First ablation | `ConnectionRegistry`, WebSocket multiplex mode, WebSocket pump tasks, forwarding-only ASGI method | 647 |
| Second ablation | HTTP memory transport pair and pump, `ConnectionState`, generic response-waiter map | 581 |
| Starlette migration | Custom ASGI app, request/header parsing, response encoding, WebSocket state tracking, `PostResult` | 483 |

This measures structural simplification and regression coverage, not throughput
or latency. Additional tests cover interrupted initialization, closing a full
SSE buffer, WebSocket cancellation/disconnect, invalid frames, and session routing
of concurrent success/error responses.

`create_asgi_app(agent_factory, *, path="/acp")` returns a Starlette application.
The default route is now `/acp`, replacing the earlier catch-all route.
`AcpServer.handle_post()` and `handle_delete()` return
Starlette responses (`status_code`, byte `body`, and case-insensitive `headers`).
`open_stream()` and `close()` keep their signatures.
The experimental `AcpAsgiApp` and `PostResult` wrappers were removed, along with
`ConnectionRegistry`, `ConnectionState`, `AcpServer.registry`, and
`create_websocket_connection()`. Direct WebSocket integrations now use
`handle_websocket(agent_factory, websocket)` with a Starlette `WebSocket`.
WebSocket lifetimes belong to their ASGI handlers; `AcpServer.close()` manages
HTTP connections.

The migration adds checks for HTTP error statuses, unsupported methods, mounting,
lifespan cleanup, and reopening an SSE stream after disconnect.

### HTTP/2 server requirement

> ⚠️ **Uvicorn does not serve HTTP/2.** For a spec-compliant Streamable HTTP
> server, run an HTTP/2-capable ASGI server (**Hypercorn**, Daphne, Granian) or
> terminate HTTP/2 at a proxy. The WebSocket profile works on Uvicorn.

```python
import asyncio
import hypercorn.asyncio
from hypercorn.config import Config

config = Config()
config.bind = ["localhost:8000"]
config.alpn_protocols = ["h2", "http/1.1"]
asyncio.run(hypercorn.asyncio.serve(app, config))
```

## Examples

- [`examples/http_server.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/http_server.py) — serve an agent over HTTP + WS (Hypercorn).
- [`examples/http_client.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/http_client.py) — connect over Streamable HTTP.
- [`examples/ws_client.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/ws_client.py) — connect over WebSocket.

## Identity model

- `Acp-Connection-Id` (HTTP header) — returned by `initialize`; required on all
  post-initialize HTTP requests and every GET stream.
- `Acp-Session-Id` (HTTP header) — required on session-scoped POSTs and the
  session-scoped GET stream.
- `sessionId` (JSON-RPC field) — carried in params/results and used to route
  messages to the correct stream.

## Not yet supported (deferred to a future revision)

- `Last-Event-ID` / SSE resumability and message sequencing.
- Client-side automatic reconnect/backoff.
- Batch JSON-RPC (the server returns `501`).
- `Acp-Protocol-Version` header enforcement.
