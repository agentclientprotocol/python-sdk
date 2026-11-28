# Quickstart

Launch a streaming ACP agent and connect a client in a few minutes (requires Python 3.10–3.15).

## 1. Install

```bash
pip install agent-client-protocol
# or
uv add agent-client-protocol
```

## 2. Run the echo agent and connect a client

`examples/echo_agent.py` is a runnable stdio agent that streams incoming prompt text back as `session/update` notifications:

```bash
python examples/echo_agent.py
```

Point any stdio ACP client (such as Zed, Toad, Lody, or T3 Code) at that command. For example, in Zed's `settings.json`:

```json
{
  "agent_servers": {
    "Echo Agent (Python)": {
      "type": "custom",
      "command": "uv",
      "args": ["run", "/abs/path/to/python-sdk/examples/echo_agent.py"]
    }
  }
}
```

## 3. Drive an agent from Python

Use `spawn_agent_process` to start an agent subprocess, exchange JSON-RPC messages over stdio, and close both cleanly on exit:

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
        await conn.prompt(session_id=session.session_id, prompt=[TextContentBlock(text="Hello from spawn!")])


asyncio.run(main())
```

`spawn_client_process` provides the matching helper when an agent spawns a client subprocess.

## 4. Write your own agent

Subclass `acp.Agent`, implement the methods your agent supports, and start serving with `acp.run_agent()`:

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

Generated schema models populate discriminator fields (`type`, `sessionUpdate`) automatically, so models like `AgentMessageChunk(content=TextContentBlock(text="Hello!"))` can be constructed directly. Extra keyword arguments on connection calls and handlers carry JSON-RPC `_meta`.

## Next steps

- Browse [`examples/`](https://github.com/agentclientprotocol/python-sdk/tree/main/examples) for fuller agent, client, HTTP/WS, MCP-over-ACP, and Gemini CLI demos.
- Continue to [Protocol Features (v1)](protocol-features.md), [Web Transport](web-transport.md), or [Experimental Protocol v2](experimental-v2.md).
