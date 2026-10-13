# Experimental Protocol v2

> **Experimental.** Protocol v2 is a draft. Import it from `acp.experimental` and
> expect its API and generated models to change with the upstream schema.

The bindings use [`schema-v2.0.0-alpha.5`](https://github.com/agentclientprotocol/agent-client-protocol/blob/schema-v2.0.0-alpha.5/schema/v2/schema.unstable.json). The official [v2 documentation](https://agentclientprotocol.com/protocol/v2/overview) describes the draft protocol; this page covers the bundled schema and Python runtime.

The v2 runtime is separate from the stable v1 API. Like v1, connection methods
and agent/client handlers accept expanded, snake-case parameters. Responses and
nested values (content blocks, updates, capabilities) use `v2.schema` models.
Extra keyword arguments carry request `_meta`.

Install update handlers before opening a session because updates are independent
connection traffic. This function opens a session on a caller-provided message
transport and returns the connection for further use. The caller closes it
during teardown. `cwd` is an absolute path on the agent's host:

```python
from typing import Any

from acp.experimental import v2


class MyClient(v2.Client):
    async def session_update(
        self, session_id: str, update: Any, **kwargs: Any,
    ) -> None:
        print(session_id, update)


async def open_session(
    transport: Any, cwd: str,
) -> tuple[v2.ClientSideConnection, str]:
    connection = v2.connect_to_agent(MyClient(), transport)
    try:
        initialized = await connection.initialize(
            protocol_version=v2.PROTOCOL_VERSION,
            info=v2.schema.Implementation(name="my-client", version="1.0.0"),
        )
        if (
            initialized.capabilities is None
            or initialized.capabilities.session is None
        ):
            raise RuntimeError("The agent does not support sessions")
        session = await connection.new_session(cwd=cwd)
    except BaseException:
        await connection.close()
        raise
    return connection, session.session_id
```

The connection validates the returned protocol version and closes on
initialization failure. Agents advertise session support with
`AgentCapabilities(session=SessionCapabilities())`; an omitted or null `session`
means unsupported. Agents supporting sessions implement the baseline methods,
including list, resume, close, prompt, cancellation, and updates.

Agent handlers use expanded parameters too. This initialization handler belongs
to an agent that implements those session methods:

```python
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
        capabilities=v2.schema.AgentCapabilities(
            session=v2.schema.SessionCapabilities()
        ),
    )
```

`v2.Agent` and `v2.Client` describe the v2 handler signatures; subclassing is
optional. Implement only the methods you support. Unimplemented requests return
method-not-found, and unimplemented notifications are ignored. The v2 protocols
are separate from v1 because initialization, prompt responses, permissions, and
session updates have different contracts. Both versions use `param_model`
metadata to derive their routes. V2 validates request and response models
and requires successful initialization before other traffic.

## Configuration and elicitation

Union requests use expanded parameters. These fragments assume an initialized
client-side `connection`, an existing `session_id`, and an agent-side
`agent_connection`. Set only options the agent exposes; request elicitation
only for modes the client advertises:

```python
await connection.set_config_option(config_id="thinking", session_id=session_id, value=True)
# type defaults to "boolean" for bool values and "id" otherwise.
await connection.set_config_option(
    config_id="vendor/limit", session_id=session_id, value=10, type="_vendor/number",
)
await agent_connection.create_elicitation(
    message="Sign in", mode="url", session_id=session_id,
    elicitation_id="sign-in-1", url="https://example.com/login",
)
```

For elicitation, `session_id` selects session scope; otherwise `request_id`
selects request scope (including `None`). Pass `requested_schema` for form mode,
or `elicitation_id` and `url` for URL mode. Handlers receive the validated
branch's fields, including `type` for config options and `mode` for elicitation.

## Prompt acceptance and completion

With the connection and session ID from `open_session`, submit a prompt:

```python
accepted = await connection.prompt(
    session_id=session_id,
    prompt=[v2.schema.TextContentBlock(text="Hello")],
)
```

`session/prompt` returns after the agent inserts the user message into the ACP
conversation, without waiting for processing to finish. The response requires a
non-null `message_id`. Agents return `v2.schema.PromptResponse(message_id=...)`
and echo the user message in a `UserMessageUpdate` or `UserMessageChunk` carrying
the same ID. That update may arrive before or after the response; use
`accepted.message_id` to match it. Other session updates are independent traffic
and do not carry a prompt identifier.

Foreground work is reported through `state_update`: `running`,
`requires_action`, or `idle`. An `IdleSessionStateUpdate` indicates that the
agent is ready for another prompt and may include `stop_reason`. Background
activity may still emit updates while idle. Track these updates in the client;
the SDK does not turn them into a completion awaitable. See the protocol's
[prompt lifecycle](https://agentclientprotocol.com/protocol/v2/prompt-lifecycle).

## Notices, patches, and replay

Agents can send `v2.schema.SessionNotice(severity="warning", title="Context is nearly full")`
with `await agent_connection.session_update(session_id=session_id, update=notice)`. V2 notices require no client capability and
are live advisory events, outside retained session history. Clients may ignore
them. Titles must be non-empty, and severity also accepts custom or future strings.

For patch fields on tool calls and terminals, omit a field to leave its current
value unchanged, or explicitly pass `None` to clear it. For example,
`v2.schema.SessionToolCallUpdate(tool_call_id="tool-1", name=None)` clears the
tool name, while omitting `name` leaves it unchanged. This also applies to
terminal updates and patch metadata. When applying received patches, use
`update.model_dump(by_alias=True, exclude_unset=True)` to retain that distinction.

Setting `replay_from=v2.schema.ReplayFromStartVariant()` on `connection.resume_session(...)`
requests all retained conversation history; agents need not retain every message.
Accepted elicitation content validates scalar values and string lists; nested
objects are not valid form values.

## Version selection and extensions

Agents that serve both versions use `AgentProtocolRouter`. In this fragment,
`V1Agent` and `V2Agent` are your application's implementations, each accepting
the selected connection in its constructor:

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

The router selects a runtime from the first initialization request and retains
it for the connection. When it selects v1 for a v2 initialization request, it
maps the initialization fields to v1. Subsequent calls use the selected runtime's
schema and handler contracts; the router does not translate between versions.

Client-side fallback is application controlled and may require opening a new
transport. Protocol-level request cancellation (`$/cancel_request`) is handled
by the shared connection layer, as in v1: cancelling the task awaiting a request
sends a best-effort cancellation to the peer, and an incoming cancellation
cancels the handler task, which answers with its result or `-32800`.
`session/cancel` remains available for cancelling active session work.
