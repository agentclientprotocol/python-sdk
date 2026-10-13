# Build an agent

Write an agent that prefixes each text prompt with `You said:`. First complete the [quickstart](quickstart.md) so you have a repository checkout and a working client.

## Implement the handlers

Save this as `my_agent.py` in the repository root:

```python
import asyncio
from typing import Any
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    Agent,
    Client,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
)
from acp.exceptions import RequestError
from acp.schema import AgentMessageChunk, ResourceContentBlock, TextContentBlock


class MyAgent(Agent):
    _client: Client

    def __init__(self) -> None:
        self._sessions: set[str] = set()
        self._active: dict[str, asyncio.Event] = {}

    def on_connect(self, conn: Client) -> None:
        self._client = conn

    async def initialize(
        self, protocol_version: int, **kwargs: Any
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=PROTOCOL_VERSION)

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        session_id = uuid4().hex
        self._sessions.add(session_id)
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self, session_id: str, prompt: list[Any], **kwargs: Any
    ) -> PromptResponse:
        if session_id not in self._sessions:
            raise RequestError.invalid_params({"sessionId": session_id})
        if session_id in self._active:
            raise RequestError.invalid_request({"details": "A turn is already active"})

        cancelled = asyncio.Event()
        self._active[session_id] = cancelled
        try:
            for block in prompt:
                if cancelled.is_set():
                    break
                if isinstance(block, TextContentBlock):
                    text = f"You said: {block.text}"
                elif isinstance(block, ResourceContentBlock):
                    text = f"You linked: {block.name} ({block.uri})"
                else:
                    raise RequestError.invalid_params(
                        {"details": "Only text and resource links are supported"}
                    )
                await self._client.session_update(
                    session_id=session_id,
                    update=AgentMessageChunk(content=TextContentBlock(text=text)),
                )
            return PromptResponse(
                stop_reason="cancelled" if cancelled.is_set() else "end_turn"
            )
        finally:
            self._active.pop(session_id, None)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        if cancelled := self._active.get(session_id):
            cancelled.set()


if __name__ == "__main__":
    asyncio.run(run_agent(MyAgent()))
```

`Agent` describes the handler interface. Subclassing it is optional; the SDK can use an object with the matching methods. The router validates prompt blocks into generated models before calling your handler. This example handles the protocol's baseline text and resource-link variants; it reports a link's name and URI without fetching the resource. Other content types require advertised prompt capabilities. The [API reference](../reference/api.md#agent-handlers) shows the complete content union.

`on_connect` receives the connection used to send callbacks to the client. `initialize` reports the supported protocol version. `new_session` returns a session ID. During `prompt`, the agent sends text through `session_update`, then returns the turn's stop reason.

`cancel` marks the session's active turn for cooperative cancellation. The prompt handler checks that flag and returns `stop_reason="cancelled"` after sending its remaining updates. The SDK dispatches the notification; your agent stops its own model and tool work. See the protocol's [prompt-turn contract](https://agentclientprotocol.com/protocol/v1/prompt-turn).

## Run it through the client

```bash
uv run --frozen python examples/client.py python my_agent.py
```

Type `Hello ACP`:

```text
> Hello ACP
| Agent: You said: Hello ACP
>
```

The client owns the agent process. Keep agent logs on stderr because stdout carries ACP frames.

## Add your own behavior

Replace the text generation inside `prompt` with your agent's work. Send multiple text chunks as results become available. Implement the agent capabilities you advertise, such as supported prompt content and session persistence. Check the client's advertised filesystem and terminal capabilities before calling those services.

This example tracks valid session IDs in memory but retains no conversation history and uses an empty MCP server list. A production runtime also handles the session's working directory and MCP server connections; the protocol requires stdio MCP support. Add session storage before advertising support for loading previous sessions. Long-running model and tool operations must observe cancellation and finish their cleanup before the prompt response. Use the [permission guide](../how-to/permissions.md) when an operation needs user approval. The [integration patterns](../integrations/index.md) show how Bub and PydanticAI adapters map an existing runtime's events and history to these handlers.
