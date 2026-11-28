<a href="https://agentclientprotocol.com/">
  <img alt="Agent Client Protocol" src="https://zed.dev/img/acp/banner-dark.webp">
</a>

# Agent Client Protocol — Python SDK

[![PyPI](https://img.shields.io/pypi/v/agent-client-protocol)](https://pypi.org/project/agent-client-protocol/)
[![Python](https://img.shields.io/pypi/pyversions/agent-client-protocol)](https://pypi.org/project/agent-client-protocol/)
[![CI](https://github.com/agentclientprotocol/python-sdk/actions/workflows/main.yml/badge.svg)](https://github.com/agentclientprotocol/python-sdk/actions/workflows/main.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://agentclientprotocol.github.io/python-sdk/)

Python SDK for building [Agent Client Protocol (ACP)](https://agentclientprotocol.com/) agents and clients, with generated Pydantic models and asyncio transports over stdio, Streamable HTTP, and WebSocket.

## Installation

```bash
pip install agent-client-protocol
# or
uv add agent-client-protocol
```

For experimental HTTP/WebSocket transports, install the `http` extra (`pip install "agent-client-protocol[http]"`). Requires Python 3.10–3.15.

## Quick example

### Build an agent

```python
import asyncio
from typing import Any
from uuid import uuid4

from acp import PROTOCOL_VERSION, Agent, InitializeResponse, NewSessionResponse, PromptResponse, run_agent
from acp.interfaces import Client
from acp.schema import AgentMessageChunk, TextContentBlock


class EchoAgent(Agent):
    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(self, protocol_version: int, **kwargs: Any) -> InitializeResponse:
        return InitializeResponse(protocol_version=PROTOCOL_VERSION)

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        return NewSessionResponse(session_id=uuid4().hex)

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        for block in prompt:
            text = getattr(block, "text", "")
            await self._conn.session_update(
                session_id=session_id,
                update=AgentMessageChunk(content=TextContentBlock(text=text)),
            )
        return PromptResponse(stop_reason="end_turn")


if __name__ == "__main__":
    asyncio.run(run_agent(EchoAgent()))
```

### Drive an agent from Python

```python
import asyncio
import sys
from typing import Any

from acp import PROTOCOL_VERSION, Client, spawn_agent_process
from acp.schema import TextContentBlock


class SimpleClient(Client):
    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        print(session_id, update)


async def main() -> None:
    async with spawn_agent_process(SimpleClient(), sys.executable, "examples/echo_agent.py") as (conn, _proc):
        await conn.initialize(protocol_version=PROTOCOL_VERSION)
        session = await conn.new_session(cwd=".", mcp_servers=[])
        await conn.prompt(session_id=session.session_id, prompt=[TextContentBlock(text="Hello ACP!")])


asyncio.run(main())
```

## Documentation

- [Quickstart](https://agentclientprotocol.github.io/python-sdk/quickstart/) — Connect your agent to Zed, Toad, Lody, or T3 Code, or drive agents programmatically.
- [Protocol Features](https://agentclientprotocol.github.io/python-sdk/protocol-features/) — Unstable opt-in methods, session notices, stateless MCP-over-ACP, cancellation, and extensions (`schema-v1.35.0`).
- [Web Transport](https://agentclientprotocol.github.io/python-sdk/web-transport/) — Serve and consume ACP over Streamable HTTP and WebSocket (`[http]`).
- [Experimental Protocol v2](https://agentclientprotocol.github.io/python-sdk/experimental-v2/) — Preview v2 runtime (`schema-v2.0.0-beta.4`) and dual-stack `AgentProtocolRouter`.
- [Examples](https://github.com/agentclientprotocol/python-sdk/tree/main/examples) · [Contrib](https://agentclientprotocol.github.io/python-sdk/contrib/) · [Use Cases](https://agentclientprotocol.github.io/python-sdk/use-cases/) · [Testing](https://agentclientprotocol.github.io/python-sdk/testing/)

## Development

```bash
make install    # Set up uv environment and pre-commit hooks
make check      # Format, lint, type-check, and dependency check
make test       # Run pytest suite
```

## Community

Join the conversation on [GitHub Discussions](https://github.com/agentclientprotocol/python-sdk/discussions), [Zulip](https://agentclientprotocol.zulipchat.com/), or [agentclientprotocol.com](https://agentclientprotocol.com/).
