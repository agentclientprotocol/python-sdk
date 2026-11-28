# Experimental Protocol v2

> **Preview.** Protocol v2 lives under `acp.experimental` and tracks [`schema-v2.0.0-beta.4`](https://github.com/agentclientprotocol/agent-client-protocol/releases/tag/schema-v2.0.0-beta.4).

Connection methods and agent/client handlers accept expanded snake_case parameters, responses and nested payloads use `acp.experimental.v2.schema` models, and extra keyword arguments serialize into `_meta`.

## Client and agent usage

Register `session_update` before opening a session so streamed updates arrive as soon as traffic starts:

```python
from typing import Any
from acp.experimental import v2


class MyClient(v2.Client):
    async def session_update(self, session_id: str, update: v2.schema.SessionUpdate, **kwargs: Any) -> None:
        handle_update(session_id, update)


connection = v2.connect_to_agent(MyClient(), transport)
await connection.initialize(
    protocol_version=v2.PROTOCOL_VERSION,
    info=v2.schema.Implementation(name="my-client", version="1.0.0"),
)
session = await connection.new_session(cwd="/workspace")
accepted = await connection.prompt(session_id=session.session_id, prompt=[v2.schema.TextContentBlock(text="Hello")])
```

Implement the handlers your agent supports (subclassing `v2.Agent` is optional) and serve with `v2.run_agent()`:

```python
from typing import Any
from acp.experimental import v2


class MyAgent(v2.Agent):
    async def initialize(
        self,
        protocol_version: int,
        info: v2.schema.Implementation,
        capabilities: v2.schema.ClientCapabilities | None = None,
        **kwargs: Any,
    ) -> v2.schema.InitializeResponse:
        return v2.schema.InitializeResponse(
            protocol_version=v2.PROTOCOL_VERSION,
            info=v2.schema.Implementation(name="my-agent", version="1.0.0"),
            capabilities=v2.schema.AgentCapabilities(session=v2.schema.SessionCapabilities()),
        )

    async def new_session(self, cwd: str, **kwargs: Any) -> v2.schema.NewSessionResponse:
        return v2.schema.NewSessionResponse(session_id="session-1")


await v2.run_agent(MyAgent())
```

V2 connections require a successful `initialize` handshake before other methods and validate requests and responses strictly.

## Key v2 behaviors

- **Asynchronous prompt submission**: `session/prompt` returns `PromptResponse(message_id=...)` once the user message is accepted, while updates stream independently. Pass `idempotency_key` (1–255 chars) so retried submissions reuse the original `message_id`.
- **Expanded union calls**: `set_config_option` infers `type` (`"boolean"` for `bool`, `"id"` otherwise, or pass a custom `type`); `create_elicitation` accepts either `session_id` or `request_id` along with form or URL parameters.
- **Patch updates**: In updates like `SessionToolCallUpdate` and `SessionTerminalUpdate`, omit a field to keep its value or pass `None` to clear it. Serialize with `update.model_dump(by_alias=True, exclude_unset=True)` to preserve that distinction.
- **Notices and replay**: Agents can emit live advisory `v2.schema.SessionNotice(...)` updates, and clients can request retained history via `resume_session(..., replay_from=v2.schema.ReplayFromStartVariant())`.
- **Reusable union aliases**: `v2.schema` exports `SessionUpdate`, `ContentBlock`, `StateUpdate`, `BackgroundWork`, and `AvailableCommand`.

## Dual-stack v1/v2 negotiation

Use `AgentProtocolRouter` to serve both protocol versions on one stdio, HTTP, or WebSocket endpoint:

```python
from acp.experimental import AgentProtocolRouter

router = AgentProtocolRouter(
    v1=lambda connection: V1Agent(connection),
    v2=lambda connection: V2Agent(connection),
)
await router.run()
```

Each connection invokes the selected factory once and runs strictly under the negotiated protocol version.

## Extensions, cancellation, and MCP-over-ACP

- Call `send_extension_request("_vendor/method", ...)` and `send_extension_notification("_vendor/event", ...)` for vendor extensions.
- `$/cancel_request` cancels running handler tasks and outgoing requests automatically; use `cancel_session(session_id=...)` for active session work.
- [Stateless MCP-over-ACP](protocol-features.md#stateless-mcp-over-acp) uses `v2.schema.McpResultResponse`, `McpErrorResponse`, and `MessageMcpResponse` when the agent advertises `capabilities.session.mcp.acp = {}`.
