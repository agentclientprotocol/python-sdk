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

This pulls in `httpx[http2]` (HTTP/2 + SSE consumption) and `websockets`.

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

The server core is framework-agnostic; a thin ASGI adapter bridges it to your
web framework:

```python
from acp.http.asgi import create_asgi_app

# One agent instance is created per connection.
app = create_asgi_app(lambda conn: MyAgent())
```

`app` is a standard ASGI 3.0 application handling `POST`/`GET`/`DELETE` and
WebSocket upgrades on the ACP endpoint.

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
