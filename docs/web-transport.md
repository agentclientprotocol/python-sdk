# Web Transport (Streamable HTTP & WebSocket)

> **Experimental.** The remote web transports are experimental and shipped under the optional `[http]` extra.

In addition to stdio, the SDK supports two remote JSON-RPC transports sharing the same ACP connection lifecycle (`initialize` → session operations → close):

- **Streamable HTTP** — `POST` carries client-to-server messages, long-lived `GET` Server-Sent Events (SSE) streams carry server-to-client messages (one connection stream plus one per session), and `DELETE` terminates the connection. `initialize` responds with `200 OK` and a JSON body; subsequent `POST` requests respond with `202 Accepted`. Requires HTTP/2.
- **WebSocket** — a `GET` WebSocket upgrade on the same endpoint carrying full-duplex JSON-RPC text frames (one message per frame). Binary frames are ignored and logged once on the server.

## Installation

```bash
pip install "agent-client-protocol[http]"
# or
uv add "agent-client-protocol[http]"
```

This extra installs `httpx2[http2]`, `websockets`, and `starlette`.

## Client usage

Both client factories return a `Transport` compatible with `connect_to_agent`:

```python
from acp import PROTOCOL_VERSION, connect_to_agent
from acp.http import create_http_stream
from acp.ws import create_websocket_stream

# Streamable HTTP
transport = create_http_stream("http://localhost:8000/acp")
conn = connect_to_agent(my_client, transport)

# WebSocket
transport = await create_websocket_stream("ws://localhost:8000/acp")
conn = connect_to_agent(my_client, transport)

await conn.initialize(protocol_version=PROTOCOL_VERSION)
session = await conn.new_session(cwd="/tmp", mcp_servers=[])
await conn.prompt(session_id=session.session_id, prompt=[...])
await conn.close()
await transport.close()
```

Over Streamable HTTP, the client sends `initialize` first, stores the returned `Acp-Connection-Id` header, and opens the connection-scoped SSE stream. When a session is created, loaded, or resumed, the client opens the corresponding session-scoped SSE stream.

### Loading and resuming sessions

Both HTTP and WebSocket support `load_session()` and `resume_session()` when advertised by the agent:

```python
init = await conn.initialize(protocol_version=PROTOCOL_VERSION)
if init.agent_capabilities and init.agent_capabilities.load_session:
    await conn.load_session(session_id="saved-session-id", cwd="/workspace", mcp_servers=[])
    await conn.prompt(session_id="saved-session-id", prompt=[...])
```

Over HTTP, `session/load` history replay and the load/resume responses stream over the connection SSE channel; once established, subsequent session traffic uses the session SSE stream. Over WebSocket, replay and live traffic share the bidirectional socket.

## Server usage

`create_asgi_app` builds a Starlette application serving `POST`/`GET`/`DELETE` and WebSocket upgrades at `/acp` by default, creating one agent instance per connection:

```python
from acp.http.asgi import create_asgi_app

app = create_asgi_app(lambda conn: MyAgent())
```

Pass `path` to change the endpoint route:

```python
app = create_asgi_app(lambda conn: MyAgent(), path="/rpc")
```

### Serving v2 or dual-stack agents

Pass an `AgentProtocolRouter` to `create_asgi_app` to negotiate v1 and v2 over both HTTP and WebSocket:

```python
from acp.experimental import AgentProtocolRouter
from acp.http.asgi import create_asgi_app

router = AgentProtocolRouter(
    v1=lambda conn: V1Agent(conn),
    v2=lambda conn: V2Agent(conn),
)
app = create_asgi_app(router)
```

### Mounting in an existing Starlette app

Enter the child lifespan from the parent application so active HTTP connections close cleanly during shutdown:

```python
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.routing import Mount

from acp.http.asgi import create_asgi_app

acp_app = create_asgi_app(lambda conn: MyAgent())


@asynccontextmanager
async def lifespan(app: Starlette):
    async with acp_app.router.lifespan_context(acp_app):
        yield


app = Starlette(routes=[Mount("/agents", app=acp_app)], lifespan=lifespan)
# Serves HTTP and WebSocket at /agents/acp
```

### HTTP/2 server setup

Streamable HTTP requires an HTTP/2-capable ASGI server such as **Hypercorn**, **Daphne**, or **Granian** (or HTTP/2 termination at a reverse proxy). WebSocket works on any ASGI server, including Uvicorn.

```python
import asyncio
import hypercorn.asyncio
from hypercorn.config import Config

config = Config()
config.bind = ["localhost:8000"]
config.alpn_protocols = ["h2", "http/1.1"]
asyncio.run(hypercorn.asyncio.serve(app, config))
```

## HTTP headers and stream routing

| Header / Field | Role |
| --- | --- |
| `Acp-Connection-Id` | Returned on `initialize`; sent on subsequent HTTP `POST`/`GET`/`DELETE` requests |
| `Acp-Session-Id` | Sent on session-scoped `POST` requests and session-scoped `GET` SSE streams |
| `sessionId` | JSON-RPC payload field used to route server-to-client messages to the matching session stream |

## Examples

- [`examples/http_server.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/http_server.py) — serve an agent over Streamable HTTP and WebSocket with Hypercorn
- [`examples/http_client.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/http_client.py) — connect over Streamable HTTP
- [`examples/ws_client.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/ws_client.py) — connect over WebSocket

## Current scope

- SSE keepalives, bounded outbound buffers, and orderly teardown on `DELETE` or server shutdown are built in.
- Reconnect/backoff policies, `Last-Event-ID` SSE resumability, and `Acp-Protocol-Version` header enforcement are planned for a future revision.
- JSON-RPC batch payloads over HTTP return `501`.
