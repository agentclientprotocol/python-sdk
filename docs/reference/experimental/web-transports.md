# Web Transport (Streamable HTTP & WebSocket)

> **Experimental.** The remote web transports are experimental and may change.
> They ship as an optional extra and are import-guarded.

The SDK can run ACP over two remote connectivity profiles in addition to stdio:

- **Streamable HTTP** — `POST` for client→server messages, long-lived `GET` SSE
  streams for server→client messages (one connection-scoped stream plus one per
  session), and `DELETE` to terminate. `initialize` returns `200 OK` with a JSON
  body; accepted post-initialization messages return `202 Accepted`. Invalid
  HTTP requests return an error status. The transport profile requires HTTP/2.
- **WebSocket** — a `GET` upgrade on the same endpoint carrying full-duplex
  JSON-RPC text frames, one message per frame. Binary frames are ignored, as
  the transport RFD specifies; the server logs a warning on the first one.

Both carry ACP JSON-RPC messages. The examples below use the v1 lifecycle
(`initialize` → session methods → close). The
[transport RFD](https://agentclientprotocol.com/rfds/streamable-http-websocket-transport)
defines these profiles; the SDK's connection and routing behavior is described below.

## Installation

```bash
pip install "agent-client-protocol[http]"
```

This pulls in `httpx2[http2]` (HTTP/2 + SSE consumption), `websockets`, and
`starlette` (the server application). The core SDK and stdio transport do not
require these optional dependencies.

## Client

Both transports produce a message transport for `connect_to_agent`. Complete
the [quickstart](../../tutorials/quickstart.md#make-the-calls-from-python) first
to create `hello_client.py`. Save this as `web_client.py` beside it:

```python
import asyncio
import sys
from pathlib import Path

from acp import PROTOCOL_VERSION, connect_to_agent
from acp.http import create_http_stream
from acp.schema import TextContentBlock
from acp.ws import create_websocket_stream

from hello_client import EchoClient


async def main(url: str, cwd: str) -> None:
    if url.startswith(("ws://", "wss://")):
        transport = await create_websocket_stream(url)
    else:
        transport = create_http_stream(url)
    conn = connect_to_agent(EchoClient(), transport)
    try:
        initialized = await conn.initialize(protocol_version=PROTOCOL_VERSION)
        if initialized.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError("The agent selected an unsupported protocol version")
        session = await conn.new_session(cwd=cwd, mcp_servers=[])
        response = await conn.prompt(
            session_id=session.session_id,
            prompt=[TextContentBlock(text="Hello ACP")],
        )
        print(f"Turn finished: {response.stop_reason}")
    finally:
        await conn.close()
        await transport.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], str(Path.cwd())))
```

Run it against the local server below with
`uv run --frozen --extra http python web_client.py ws://localhost:8000/acp`.
It prints `You said: Hello ACP` and `Turn finished: end_turn`. For a remote
agent, pass its absolute workspace path as `cwd` instead of a local path.

For Streamable HTTP, the client sends `initialize` first, reads the `Acp-Connection-Id` response
header, then opens the connection-scoped SSE stream. When a new `sessionId`
appears it opens that session-scoped stream too. A single SSE attempt is made per
stream; reconnect/retry is the caller's responsibility. Losing the connection
stream ends the connection's receive loop and rejects outstanding requests.

### Loading an existing session

Both HTTP and WebSocket support `load_session()` when the agent advertises
`loadSession` and implements session persistence:

Within the `try` block above, replace `new_session` with a load call after
initialization. `saved_session_id` must identify a session retained by the
agent, and `cwd` must match that session's working directory:

```python
if (
    initialized.agent_capabilities is None
    or not initialized.agent_capabilities.load_session
):
    raise RuntimeError("The agent does not support loading sessions")
await conn.load_session(session_id=saved_session_id, cwd=cwd, mcp_servers=[])
response = await conn.prompt(
    session_id=saved_session_id,
    prompt=[TextContentBlock(text="Continue")],
)
```

For HTTP, history replay and the load response use the connection SSE stream.
The client correlates the response with the session ID from the load request,
then opens the session SSE stream for further prompts and agent callbacks. This
also works with an empty history: a load response need not contain `sessionId`.
Replay is consumed as it arrives, so histories larger than the SSE buffer do not
wait for a session stream to open. WebSocket uses its existing bidirectional
connection for both replay and subsequent messages.

`resume_session()` (when the agent advertises `sessionCapabilities.resume`)
follows the same HTTP routing without replaying history: the response uses the
connection SSE stream, and the client then opens the session SSE stream.

A failed load returns its JSON-RPC error on the connection stream and can be
retried. The server removes streams provisioned only for failed loads, while
preserving established sessions and overlapping loads. It does not change the
agent's load response or automatically enable the agent's `loadSession` capability.

## Server

The server uses Starlette for HTTP requests, responses, routing, streaming,
WebSocket handling, and application lifespan. Save the agent from
[Build an agent](../../tutorials/agent.md) as `my_agent.py`, then create
`web_server.py` beside it:

```python
from acp.http.asgi import create_asgi_app

from my_agent import MyAgent

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

For the connection registry and stream routing, see [Transport architecture](../../explanation/transports.md).

### Server API

The experimental server API uses Starlette. `create_asgi_app(agent_factory, *, path="/acp")`
returns a Starlette application, and `AcpServer.handle_post()` and `handle_delete()`
return Starlette response objects. `open_stream()` yields SSE bytes and `close()`
closes active connections.

Direct WebSocket integrations use
`handle_websocket(agent_factory, websocket)` with a Starlette `WebSocket`.
The default route is `/acp`; pass `path="/"` explicitly if you need the mount root.

### HTTP/2 server requirement

The Streamable HTTP profile requires HTTP/2 for multiplexing POST requests and
SSE streams. Serve HTTP/2 with Hypercorn and TLS, or terminate HTTP/2 at a proxy.
The SDK enables HTTP/2 on its default client but does not reject HTTP/1.1
responses. Plain local HTTP can therefore exercise the SDK without satisfying
the profile's HTTP/2 requirement. WebSocket does not need an HTTP/2 server.

Start the local WebSocket server with:

```bash
uv run --frozen --extra http --with hypercorn hypercorn web_server:app --bind localhost:8000
```

For HTTP/2 over TLS, supply `--certfile` and `--keyfile` for a certificate
trusted by your client, then use an `https://` URL. See Hypercorn's
[HTTP/2 guidance](https://hypercorn.readthedocs.io/en/latest/discussion/http2.html).

## Identity model

- `Acp-Connection-Id` (HTTP header) — returned by `initialize`; required on all
  post-initialize HTTP requests and every GET stream.
- `Acp-Session-Id` (HTTP header) — required on session-scoped POSTs and the
  session-scoped GET stream.
- `sessionId` (JSON-RPC field) — carried in params/results and used to route
  messages to the correct stream.

The connection and session headers identify routing state; they are not
authentication credentials. Authentication and authorization belong to the
application serving the endpoint.

## Unsupported transport features

- `Last-Event-ID` / SSE resumability and message sequencing.
- Client-side automatic reconnect/backoff.
- Batch JSON-RPC (the server returns `501`).
- `Acp-Protocol-Version` header enforcement.
