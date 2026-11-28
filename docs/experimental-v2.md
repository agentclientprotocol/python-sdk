# Experimental Protocol v2

> **Preview.** Protocol v2 lives under `acp.experimental` and tracks upstream beta releases. Its API and generated models evolve alongside the v2 specification.

The v2 bindings target [`schema-v2.0.0-beta.4`](https://github.com/agentclientprotocol/agent-client-protocol/releases/tag/schema-v2.0.0-beta.4). Connection methods and agent/client handlers accept expanded snake_case parameters, responses and nested payloads use `acp.experimental.v2.schema` models, and extra keyword arguments serialize into `_meta`.

## Client usage

Register your `session_update` handler before opening a session so incoming updates are captured as soon as session traffic starts:

```python
from typing import Any
from acp.experimental import v2


class MyClient(v2.Client):
    async def session_update(
        self,
        session_id: str,
        update: v2.schema.SessionUpdate,
        **kwargs: Any,
    ) -> None:
        handle_update(session_id, update)


connection = v2.connect_to_agent(MyClient(), transport)
initialized = await connection.initialize(
    protocol_version=v2.PROTOCOL_VERSION,
    info=v2.schema.Implementation(name="my-client", version="1.0.0"),
)
session = await connection.new_session(cwd="/workspace")
accepted = await connection.prompt(
    session_id=session.session_id,
    prompt=[v2.schema.TextContentBlock(text="Hello")],
)
```

## Agent usage

Implement the methods your agent supports (subclassing `v2.Agent` is optional) and start serving with `v2.run_agent()`:

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

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> v2.schema.NewSessionResponse:
        return v2.schema.NewSessionResponse(session_id="session-1")


await v2.run_agent(MyAgent())
```

Unimplemented request handlers return JSON-RPC method-not-found (`-32601`), and unimplemented notifications are ignored. V2 connections require a successful `initialize` handshake before processing other methods and enforce strict request/response validation.

## Union requests

Config options and elicitation requests use expanded parameters:

```python
# `type` defaults to "boolean" for bool values and "id" otherwise.
await connection.set_config_option(config_id="thinking", session_id=session.session_id, value=True)
await connection.set_config_option(
    config_id="vendor/limit",
    session_id=session.session_id,
    value=10,
    type="vendor/number",
)

await agent_connection.create_elicitation(
    message="Sign in",
    mode="url",
    session_id=session.session_id,
    elicitation_id="sign-in-1",
    url="https://example.com/login",
)
```

For `create_elicitation`, pass `session_id` for session scope or a string/integer `request_id` for request scope. Pass `requested_schema` for form mode, or `elicitation_id` and `url` for URL mode. Accepted elicitation content preserves scalars, string lists, and custom JSON values.

## Prompts, updates, notices, and replay

- **Prompt submission**: `session/prompt` returns `PromptResponse(message_id=...)` once the user message is accepted into the conversation, allowing independent streaming updates. Pass `idempotency_key` (1–255 characters) so retried submissions return the original `message_id`. Agents echo the user message in a `UserMessageUpdate` or `UserMessageChunk` carrying `accepted.message_id`.
- **Session notices**: Agents can emit live advisory events via `v2.schema.SessionNotice(severity="warning", title="Context is nearly full")` with `agent_connection.session_update(...)`. Titles must be non-empty; severity accepts standard and custom string values.
- **Patch semantics**: In patch updates such as `SessionToolCallUpdate` and `SessionTerminalUpdate`, omit a field to leave its current value unchanged, or pass `None` to clear it. Call `update.model_dump(by_alias=True, exclude_unset=True)` to preserve that distinction when applying patches.
- **Replay on resume**: Pass `replay_from=v2.schema.ReplayFromStartVariant()` to `connection.resume_session(...)` to request retained conversation history.
- **Reusable union aliases**: `v2.schema` exports `SessionUpdate`, `ContentBlock`, `StateUpdate`, `BackgroundWork`, and `AvailableCommand` for type annotations and validation.

## Dual-stack v1/v2 negotiation

Use `AgentProtocolRouter` to serve both protocol versions on the same endpoint (stdio, HTTP, or WebSocket):

```python
from acp.experimental import AgentProtocolRouter

router = AgentProtocolRouter(
    v1=lambda connection: V1Agent(connection),
    v2=lambda connection: V2Agent(connection),
)
await router.run()
```

The selected factory runs once per connection and should return a fresh agent instance. When an agent configured with v1 receives a v2 `initialize` request, the router translates the common initialization fields and negotiates down to v1; once negotiated, the connection enforces the selected protocol version strictly.

## Extension methods and cancellation

Extension method names carry the protocol-required `_` prefix:

```python
result = await connection.send_extension_request("_vendor/method", {"value": 1})
await connection.send_extension_notification("_vendor/event", {"value": 1})
```

Protocol-level request cancellation (`$/cancel_request`) cancels the running handler task (answering with its return value if caught, or `-32800` otherwise), and cancelling a task awaiting an outgoing request sends a best-effort `$/cancel_request`. Use `connection.cancel_session(session_id=...)` to stop active session work.

## Stateless MCP-over-ACP

V2 uses the same [stateless MCP binding](protocol-features.md#stateless-mcp-over-acp) with `v2.schema.McpResultResponse`, `McpErrorResponse`, `McpError`, and `MessageMcpResponse`. Agents advertise support via `capabilities.session.mcp.acp = {}`, send `mcp_message` requests with explicit `server_id` and `request_id` strings, and receive provider notifications through `notify_mcp`.
