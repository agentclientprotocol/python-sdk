<a href="https://agentclientprotocol.com/">
  <img alt="Agent Client Protocol" src="https://zed.dev/img/acp/banner-dark.webp">
</a>

# Agent Client Protocol — Python SDK

[![PyPI](https://img.shields.io/pypi/v/agent-client-protocol)](https://pypi.org/project/agent-client-protocol/)
[![Python](https://img.shields.io/pypi/pyversions/agent-client-protocol)](https://pypi.org/project/agent-client-protocol/)
[![CI](https://github.com/agentclientprotocol/python-sdk/actions/workflows/main.yml/badge.svg)](https://github.com/agentclientprotocol/python-sdk/actions/workflows/main.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://agentclientprotocol.github.io/python-sdk/)

Python SDK for building [Agent Client Protocol (ACP)](https://agentclientprotocol.com/) agents and clients with generated Pydantic v2 models, asyncio JSON-RPC transports, and runnable examples.

## Installation

```bash
pip install agent-client-protocol
# or
uv add agent-client-protocol
```

Optional extras:

```bash
pip install "agent-client-protocol[http]"     # Streamable HTTP and WebSocket transports (experimental)
pip install "agent-client-protocol[logfire]"  # Logfire / OpenTelemetry instrumentation
```

Requires Python 3.10–3.15.

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

### Connect from a client

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

## Capabilities and documentation

- **ACP v1 (`acp`, `acp.schema`)** — Stable bindings for `schema-v1.35.0` with stdio JSON-RPC framing (`run_agent`, `connect_to_agent`, `spawn_agent_process`). Read the [Quickstart](https://agentclientprotocol.github.io/python-sdk/quickstart/) and [Protocol Features](https://agentclientprotocol.github.io/python-sdk/protocol-features/).
- **Experimental ACP v2 (`acp.experimental.v2`)** — Preview bindings for `schema-v2.0.0-beta.4` and dual-stack version negotiation via `AgentProtocolRouter`. Read [Experimental Protocol v2](https://agentclientprotocol.github.io/python-sdk/experimental-v2/).
- **Web transports (`acp.http`, `acp.ws`)** — Experimental Streamable HTTP (SSE + HTTP/2) and WebSocket client/server support via the `[http]` extra. Read [Web Transport](https://agentclientprotocol.github.io/python-sdk/web-transport/).
- **Stateless MCP-over-ACP** — Typed `mcp/message` request and notification envelopes across stdio, HTTP, and WebSocket. Read [MCP-over-ACP](https://agentclientprotocol.github.io/python-sdk/protocol-features/#stateless-mcp-over-acp).
- **Contrib utilities (`acp.contrib`)** — Experimental `SessionAccumulator`, `ToolCallTracker`, and `PermissionBroker` helpers. Read [Contrib Utilities](https://agentclientprotocol.github.io/python-sdk/contrib/).

More resources: [Documentation Hub](https://agentclientprotocol.github.io/python-sdk/) · [Examples](https://github.com/agentclientprotocol/python-sdk/tree/main/examples) · [Use Cases](https://agentclientprotocol.github.io/python-sdk/use-cases/) · [Testing & Conformance](https://agentclientprotocol.github.io/python-sdk/testing/) · [Releasing](https://agentclientprotocol.github.io/python-sdk/releasing/)

## Development

```bash
make install     # Create uv environment and install pre-commit hooks
make check       # Lockfile check, Ruff lint/format, ty type check, deptry
make test        # Run pytest with doctests
make gen-check   # Verify generated schema bindings match schema/
make docs-test   # Build MkDocs site strictly
```

Regenerate bindings when upstream schemas advance with `ACP_SCHEMA_VERSION=<ref> make gen-all` (or `make gen-v2` for experimental v2).

## Community

- Issues and pull requests: <https://github.com/agentclientprotocol/python-sdk>
- Discussions: <https://github.com/agentclientprotocol/python-sdk/discussions>
- ACP specification and ecosystem: <https://agentclientprotocol.com/>
- Zulip chat: <https://agentclientprotocol.zulipchat.com/>
