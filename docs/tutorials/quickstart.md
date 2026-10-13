# Run an agent and client

Send a message from a Python client to an echo agent and receive a streamed reply. Both processes run locally over stdio. You need Python 3.10 through 3.14, Git, and uv.

This tutorial uses the SDK in the repository checkout. It does not require an editor, an AI model, or an API key.

## Get the examples

```bash
git clone https://github.com/agentclientprotocol/python-sdk.git
cd python-sdk
uv sync --frozen
```

Run the remaining commands from `python-sdk/`. The repository includes the examples and a lockfile for their dependencies.

## Send your first message

```bash
uv run --frozen python examples/client.py python examples/echo_agent.py
```

The client starts the agent as a child process, initializes the connection, and creates a session. At the prompt, type `Hello ACP`:

```text
> Hello ACP
| Agent: Hello ACP
>
```

The `| Agent:` line comes from a `session/update` notification. After the agent finishes the turn, the client prompts for another message. Press Ctrl+D on Unix, or Ctrl+Z followed by Enter on Windows, to exit.

Starting `echo_agent.py` by itself leaves it waiting for protocol messages on stdin. The client command above creates the pipes and sends those messages.

<span id="programmatic-launch"></span>

## Make the calls from Python

Save the following as `hello_client.py` in the repository root:

```python
import asyncio
import sys
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, Client, spawn_agent_process
from acp.schema import (
    AgentMessageChunk,
    DeniedOutcome,
    PermissionOption,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallUpdate,
)


class EchoClient(Client):
    async def session_update(
        self, session_id: str, update: Any, **kwargs: Any
    ) -> None:
        if isinstance(update, AgentMessageChunk) and isinstance(
            update.content, TextContentBlock
        ):
            print(update.content.text)

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        return RequestPermissionResponse(outcome=DeniedOutcome())


async def main() -> None:
    root = Path(__file__).resolve().parent
    agent_script = root / "examples" / "echo_agent.py"
    async with spawn_agent_process(
        EchoClient(), sys.executable, str(agent_script), cwd=root
    ) as (connection, _process):
        initialized = await connection.initialize(protocol_version=PROTOCOL_VERSION)
        if initialized.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError("The agent selected an unsupported protocol version")
        session = await connection.new_session(cwd=str(root), mcp_servers=[])
        response = await connection.prompt(
            session_id=session.session_id,
            prompt=[TextContentBlock(text="Hello ACP")],
        )
        print(f"Turn finished: {response.stop_reason}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it with the same environment as the agent:

```bash
uv run --frozen python hello_client.py
```

Expected output:

```text
Hello ACP
Turn finished: end_turn
```

`initialize` negotiates the protocol version; this client continues only when the agent selects v1. `new_session` creates a session with an absolute working directory. `prompt` sends content to that session. The agent sends response text to `session_update` and then returns a `PromptResponse` with the turn's stop reason. See the protocol's [initialization contract](https://agentclientprotocol.com/protocol/v1/initialization).

The callback accepts every update variant, but this example prints only agent text chunks. The echo agent does not request permissions; the permission handler returns a typed cancellation outcome if a request arrives. The `async with` block closes the connection and shuts down the child process when it exits. The [lifecycle explanation](../explanation/lifecycle.md) describes how these calls fit together.

## If the example does not work

| Symptom | Check |
| --- | --- |
| `examples/echo_agent.py` cannot be found | Use the cloned repository and save `hello_client.py` in its root |
| `ModuleNotFoundError: acp` | Run `uv sync --frozen`, then use `uv run --frozen python` |
| The agent waits without replying | Start it through the client; a standalone agent expects protocol messages on stdin |
| JSON-RPC parsing errors after adding logging | Send agent logs to stderr; stdout carries protocol frames |

## Extend the agent

Read [`examples/echo_agent.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/echo_agent.py) for the text round trip. The [agent tutorial](agent.md) adds resource-link handling, session tracking, and turn cancellation to that structure.

Follow [Build an agent](agent.md) to write your own handlers, or [Build a client](client.md) to implement callbacks and manage connections. For tool calls, see [Handle permissions](../how-to/permissions.md). The [integration patterns](../integrations/index.md) show how existing projects use the same SDK interfaces.

## Protocol and maintenance topics

- <span id="session-notices-unstable"></span>[Session notices](../reference/protocol.md#session-notices-unstable)
- <span id="additional-v1-protocol-methods"></span>[Additional v1 methods](../reference/protocol.md#additional-v1-protocol-methods)
- <span id="request-cancellation"></span>[Request cancellation](../reference/protocol.md#request-cancellation)
- <span id="maintaining-protocol-routes"></span>[Protocol route maintenance](../maintainers/protocol-routes.md)
- <span id="optional-talk-to-the-gemini-cli"></span>[Existing client examples](client.md#existing-client-code)
