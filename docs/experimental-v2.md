# Experimental Protocol v2

> **Experimental.** Protocol v2 is a draft. Import it from `acp.experimental` and
> expect its API and generated models to change with the upstream schema.

The bindings use `schema-v2.0.0-alpha.5`.

The v2 runtime is separate from the stable v1 API. Its methods accept and return
generated request and response models directly. Install update handlers on the
client before opening a session because updates are independent connection
traffic:

```python
from acp.experimental import v2

class MyClient:
    async def session_update(
        self,
        notification: v2.schema.UpdateSessionNotification,
    ) -> None:
        handle_update(notification)


connection = v2.connect_to_agent(MyClient(), transport)
initialized = await connection.initialize(
    v2.schema.InitializeRequest(
        protocol_version=v2.PROTOCOL_VERSION,
        info=v2.schema.Implementation(name="my-client", version="1.0.0"),
    )
)
session = await connection.new_session(
    v2.schema.NewSessionRequest(cwd="/workspace")
)
accepted = await connection.prompt(
    v2.schema.PromptRequest(
        session_id=session.session_id,
        prompt=[v2.schema.TextContentBlock(text="Hello")],
    )
)
```

`session/prompt` returns after the agent inserts the user message into the ACP
conversation, without waiting for processing to finish. The response requires a
non-null `message_id`. Agents return `v2.schema.PromptResponse(message_id=...)`
and echo the user message in a `UserMessageUpdate` or `UserMessageChunk` carrying
the same ID. That update may arrive before or after the response; use
`accepted.message_id` to match it. Other session updates are independent traffic
and do not carry a prompt identifier.

Agents can send `v2.schema.SessionNotice(severity="warning", title="Context is nearly full")`
in an `UpdateSessionNotification`. V2 notices require no client capability and
are live advisory events, outside retained session history. Clients may ignore
them. Titles must be non-empty, and severity also accepts custom or future strings.

For patch fields in session updates, omit a field to leave its current
value unchanged, or explicitly pass `None` to clear it. For example,
`v2.schema.SessionToolCallUpdate(tool_call_id="tool-1", name=None)` clears the
tool name, while omitting `name` leaves it unchanged. This also applies to
terminal updates and patch metadata. When applying received patches, use
`update.model_dump(by_alias=True, exclude_unset=True)` to retain that distinction.

Setting `replay_from=v2.schema.ReplayFromStartVariant()` on a `ResumeSessionRequest`
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
