# Experimental Protocol v2

> **Experimental.** Protocol v2 is a draft. Import it from `acp.experimental` and
> expect its API and generated models to change with the upstream schema.

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
await connection.prompt(
    v2.schema.PromptRequest(
        session_id=session.session_id,
        prompt=[v2.schema.TextContentBlock(text="Hello")],
    )
)
```

`session/prompt` returns when the agent accepts the prompt. It does not define a
boundary for session updates: they may arrive before, during, or after that
request, and they do not carry a prompt identifier. Applications decide how to
buffer or present them.

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
