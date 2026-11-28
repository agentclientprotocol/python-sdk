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
# Streamable HTTP and WebSocket transports (experimental)
pip install "agent-client-protocol[http]"

# Logfire / OpenTelemetry instrumentation
pip install "agent-client-protocol[logfire]"
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

## Protocol and transport support

| Capability | Import / Extra | Status | Reference |
| --- | --- | --- | --- |
| ACP v1 (`schema-v1.35.0`) | `acp`, `acp.schema` | Stable | [Quickstart](https://agentclientprotocol.github.io/python-sdk/quickstart/) · [Protocol Features](https://agentclientprotocol.github.io/python-sdk/protocol-features/) |
| ACP v2 (`schema-v2.0.0-beta.4`) + dual-stack router | `acp.experimental.v2`, `acp.experimental.AgentProtocolRouter` | Preview | [Experimental v2](https://agentclientprotocol.github.io/python-sdk/experimental-v2/) |
| Stdio JSON-RPC (NDJSON) | Built-in (`run_agent`, `connect_to_agent`, `spawn_agent_process`) | Stable | [Quickstart](https://agentclientprotocol.github.io/python-sdk/quickstart/) |
| Streamable HTTP & WebSocket | `acp.http`, `acp.ws` (`[http]`) | Experimental | [Web Transport](https://agentclientprotocol.github.io/python-sdk/web-transport/) |
| Stateless MCP-over-ACP | `acp` / `acp.experimental.v2` | Unstable / Preview | [Protocol Features](https://agentclientprotocol.github.io/python-sdk/protocol-features/#stateless-mcp-over-acp) |
| Session, tool call, and permission helpers | `acp.contrib` | Experimental | [Contrib Utilities](https://agentclientprotocol.github.io/python-sdk/contrib/) |

## Documentation

| Topic | Link |
| --- | --- |
| Documentation hub | <https://agentclientprotocol.github.io/python-sdk/> |
| Quickstart | <https://agentclientprotocol.github.io/python-sdk/quickstart/> |
| Protocol features (v1) | <https://agentclientprotocol.github.io/python-sdk/protocol-features/> |
| Web transport (HTTP & WebSocket) | <https://agentclientprotocol.github.io/python-sdk/web-transport/> |
| Experimental protocol v2 | <https://agentclientprotocol.github.io/python-sdk/experimental-v2/> |
| Contrib utilities | <https://agentclientprotocol.github.io/python-sdk/contrib/> |
| Use cases | <https://agentclientprotocol.github.io/python-sdk/use-cases/> |
| Testing & conformance | <https://agentclientprotocol.github.io/python-sdk/testing/> |
| Releasing | <https://agentclientprotocol.github.io/python-sdk/releasing/> |
| Runnable examples | <https://github.com/agentclientprotocol/python-sdk/tree/main/examples> |

## Development

```bash
make install     # Create uv environment and install pre-commit hooks
make check       # Lockfile check, Ruff lint/format, ty type check, deptry
make test        # Run pytest with doctests
make gen-check   # Verify generated schema bindings match schema/
make docs-test   # Build MkDocs site strictly
```

To regenerate protocol bindings when upstream schemas advance, run `ACP_SCHEMA_VERSION=<ref> make gen-all` (or `make gen-v2` for v2).

## Community

- Issues and pull requests: <https://github.com/agentclientprotocol/python-sdk>
- Discussions: <https://github.com/agentclientprotocol/python-sdk/discussions>
- ACP specification and ecosystem: <https://agentclientprotocol.com/>
- Zulip chat: <https://agentclientprotocol.zulipchat.com/>
