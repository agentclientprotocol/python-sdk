# Experimental Protocol v2

> **Experimental.** Protocol v2 is a draft. Import it from `acp.experimental` and
> expect its API and generated models to change with the upstream schema.

The bindings use `schema-v2.0.0-alpha.5`.

The v2 runtime is separate from the stable v1 API. Like v1, connection methods
and agent/client handlers accept expanded, snake-case parameters. Responses and
nested values (content blocks, updates, capabilities) use `v2.schema` models.
Extra keyword arguments carry request `_meta`.

Install update handlers before opening a session because updates are independent
connection traffic:

```python
from typing import Any
from acp.experimental import v2

class MyClient(v2.Client):
    async def session_update(
        self, session_id: str, update: Any, **kwargs: Any,
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

Implement an agent with the same expanded handler style:

```python
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

`v2.Agent` and `v2.Client` describe the v2 handler signatures; subclassing is
optional. Implement only the methods you support. Unimplemented requests return
method-not-found, and unimplemented notifications are ignored. The v2 protocols
are separate from v1 because initialization, prompt responses, permissions, and
session updates have different contracts. Both versions use `param_model`
metadata to derive their routes. V2 retains strict request/response validation
and requires successful initialization before other traffic.

Previously, v2 methods accepted a whole request model. Replace
`connection.new_session(v2.schema.NewSessionRequest(cwd="/workspace"))` with
`connection.new_session(cwd="/workspace")`, and expand handler parameters likewise.

Union requests also use expanded parameters:

```python
await connection.set_config_option(config_id="thinking", session_id=session.session_id, value=True)
# type defaults to "boolean" for bool values and "id" otherwise.
await connection.set_config_option(
    config_id="vendor/limit", session_id=session.session_id, value=10, type="vendor/number",
)
await agent_connection.create_elicitation(
    message="Sign in", mode="url", session_id=session.session_id,
    elicitation_id="sign-in-1", url="https://example.com/login",
)
```

For elicitation, `session_id` selects session scope; otherwise `request_id`
selects request scope (including `None`). Pass `requested_schema` for form mode,
or `elicitation_id` and `url` for URL mode. Handlers receive the validated
branch's fields, including `type` for config options and `mode` for elicitation.

`session/prompt` returns after the agent inserts the user message into the ACP
conversation, without waiting for processing to finish. The response requires a
non-null `message_id`. Agents return `v2.schema.PromptResponse(message_id=...)`
and echo the user message in a `UserMessageUpdate` or `UserMessageChunk` carrying
the same ID. That update may arrive before or after the response; use
`accepted.message_id` to match it. Other session updates are independent traffic
and do not carry a prompt identifier.

Agents can send `v2.schema.SessionNotice(severity="warning", title="Context is nearly full")`
with `await agent_connection.session_update(session_id=session_id, update=notice)`. V2 notices require no client capability and
are live advisory events, outside retained session history. Clients may ignore
them. Titles must be non-empty, and severity also accepts custom or future strings.

For patch fields in session updates, omit a field to leave its current
value unchanged, or explicitly pass `None` to clear it. For example,
`v2.schema.SessionToolCallUpdate(tool_call_id="tool-1", name=None)` clears the
tool name, while omitting `name` leaves it unchanged. This also applies to
terminal updates and patch metadata. When applying received patches, use
`update.model_dump(by_alias=True, exclude_unset=True)` to retain that distinction.

Setting `replay_from=v2.schema.ReplayFromStartVariant()` on `connection.resume_session(...)`
requests all retained conversation history; agents need not retain every message.
Accepted elicitation content validates scalar values and string lists; nested
objects are not valid form values.

Agents that serve both versions use `AgentProtocolRouter`:

```python
from acp.experimental import AgentProtocolRouter

router = AgentProtocolRouter(
    v1=lambda connection: V1Agent(connection),
    v2=lambda connection: V2Agent(connection),
)
await router.run()
```

The selected factory is called once per connection. Return a fresh agent from
each call to avoid sharing connection state.

Extension method names are explicit and must include the protocol-required `_`
prefix:

```python
result = await connection.send_extension_request("_vendor/method", {"value": 1})
await connection.send_extension_notification("_vendor/event", {"value": 1})
```

The selected runtime remains strict after initialization: v1 messages are not
accepted by a v2 connection, and v2 messages are not translated into v1 calls.
Only the initial v2 request is reduced to the common v1 initialization fields
when an agent selects v1.

Client-side fallback is application controlled and may require opening a new
transport. Protocol-level request cancellation is not yet exposed by the
experimental runtime; `session/cancel` remains available for cancelling active
session work.
