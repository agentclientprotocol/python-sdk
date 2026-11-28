# Quickstart

Launch a working ACP agent and client in minutes.

## Prerequisites

- Python 3.10–3.15 with `pip` or `uv`
- An ACP client (such as Zed, Toad, Lody, or T3 Code) or the SDK's programmatic client helpers

## 1. Install the SDK

```bash
pip install agent-client-protocol
# or
uv add agent-client-protocol
```

## 2. Run the echo agent

`examples/echo_agent.py` implements a minimal streaming agent over stdio:

```bash
python examples/echo_agent.py
```

The process reads newline-delimited JSON-RPC frames from `stdin` and writes responses and `session/update` notifications to `stdout`.

## 3. Connect from an ACP client

### Zed

Add an entry to Zed's `settings.json`:

```json
{
  "agent_servers": {
    "Echo Agent (Python)": {
      "type": "custom",
      "command": "uv",
      "args": [
        "run",
        "/abs/path/to/python-sdk/examples/echo_agent.py"
      ]
    }
  }
}
```

Open the Agents panel, select **Echo Agent (Python)**, and send a message to observe streamed `session/update` responses.

### Other stdio clients

Any ACP-compatible stdio client (including Toad, Lody, and T3 Code) can launch the same command and arguments directly.

### Programmatic launch

Use `spawn_agent_process` to start an agent subprocess, attach stdio framing, and close both the connection and subprocess on exit:

```python
import asyncio
import sys
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, Client, spawn_agent_process
from acp.schema import TextContentBlock


class SimpleClient(Client):
    async def request_permission(self, session_id: str, tool_call: Any, options: list[Any], **kwargs: Any) -> Any:
        return {"outcome": {"outcome": "cancelled"}}

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        print("update:", session_id, update)


async def main() -> None:
    script = Path("examples/echo_agent.py")
    async with spawn_agent_process(SimpleClient(), sys.executable, str(script)) as (conn, _proc):
        await conn.initialize(protocol_version=PROTOCOL_VERSION)
        session = await conn.new_session(cwd=str(script.parent), mcp_servers=[])
        await conn.prompt(
            session_id=session.session_id,
            prompt=[TextContentBlock(text="Hello from spawn!")],
        )


asyncio.run(main())
```

`spawn_client_process` provides the corresponding helper for spawning an ACP client subprocess from an agent.

## 4. Build a custom agent

Subclass `acp.Agent`, implement the methods your agent supports, and start the stdio loop with `acp.run_agent()`:

```python
import asyncio
from typing import Any
from uuid import uuid4

from acp import PROTOCOL_VERSION, Agent, InitializeResponse, NewSessionResponse, PromptResponse, run_agent
from acp.interfaces import Client
from acp.schema import AgentMessageChunk, TextContentBlock


class MyAgent(Agent):
    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(self, protocol_version: int, **kwargs: Any) -> InitializeResponse:
        return InitializeResponse(protocol_version=PROTOCOL_VERSION)

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        return NewSessionResponse(session_id=uuid4().hex)

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        await self._conn.session_update(
            session_id=session_id,
            update=AgentMessageChunk(content=TextContentBlock(text="Hello!")),
        )
        return PromptResponse(stop_reason="end_turn")


if __name__ == "__main__":
    asyncio.run(run_agent(MyAgent()))
```

Generated schema models provide defaults for discriminator fields such as `type` and `sessionUpdate`, enabling direct construction:

```python
from acp.schema import AgentMessageChunk, TextContentBlock

update = AgentMessageChunk(content=TextContentBlock(text="File opened."))
```

Handlers and connection methods accept expanded snake_case parameters, with additional keyword arguments serialized as request `_meta`.

## Runnable examples

- [`examples/echo_agent.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/echo_agent.py) — minimal streaming agent
- [`examples/agent.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/agent.py) — agent with capability negotiation and richer updates
- [`examples/client.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/client.py) — interactive stdio client
- [`examples/duet.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/duet.py) — in-process client/agent pairing with `spawn_agent_process`
- [`examples/mcp_over_acp.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/mcp_over_acp.py) — stateless MCP-over-ACP exchange
- [`examples/gemini.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/gemini.py) — bridge to the Gemini CLI (`gemini --acp`)
- [`examples/http_server.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/http_server.py), [`examples/http_client.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/http_client.py), [`examples/ws_client.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/ws_client.py) — Streamable HTTP and WebSocket demos

## Next steps

- Explore v1 capabilities in [Protocol Features (v1)](protocol-features.md).
- Serve agents remotely with [Web Transport (HTTP & WebSocket)](web-transport.md).
- Try the preview protocol in [Experimental Protocol v2](experimental-v2.md).
- Reuse stateful UI/agent helpers from [Contrib Utilities](contrib.md).
- Validate integrations with [Testing & Conformance](testing.md).
