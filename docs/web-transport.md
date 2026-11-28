# Web Transport (Streamable HTTP & WebSocket)

> **Experimental.** Shipped under the optional `[http]` extra.

Run ACP remotely over **Streamable HTTP** (`POST` requests + HTTP/2 SSE `GET` streams + `DELETE` teardown) or **WebSocket** (full-duplex JSON-RPC text frames) using the same `Agent` and `Client` APIs.

## Installation

```bash
pip install "agent-client-protocol[http]"
```

## Client usage

Pass a Streamable HTTP or WebSocket transport into `connect_to_agent`:

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

Both transports also support `load_session()` and `resume_session()` when advertised by the agent, streaming replay updates over the connection channel before switching to live session traffic.

## Server usage

`create_asgi_app` creates a Starlette app serving both Streamable HTTP and WebSocket at `/acp` (or a custom `path`), instantiating one agent per connection:

```python
from acp.http.asgi import create_asgi_app

app = create_asgi_app(lambda conn: MyAgent())
```

Pass an `AgentProtocolRouter` to serve v2 or negotiate v1/v2 over the same endpoint:

```python
from acp.experimental import AgentProtocolRouter
from acp.http.asgi import create_asgi_app

app = create_asgi_app(AgentProtocolRouter(v1=lambda conn: V1Agent(conn), v2=lambda conn: V2Agent(conn)))
```

When mounting `app` inside another Starlette application, enter `acp_app.router.lifespan_context(acp_app)` from the parent lifespan so active HTTP connections close cleanly on shutdown.

### Running with HTTP/2

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

See [`examples/http_server.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/http_server.py), [`examples/http_client.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/http_client.py), and [`examples/ws_client.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/ws_client.py) for complete runnable scripts.
