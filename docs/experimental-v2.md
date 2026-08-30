# Experimental Protocol v2

> **Experimental.** Protocol v2 is a draft. Import it from `acp.experimental` and
> expect its API and generated models to change with the upstream schema.

The v2 runtime is separate from the stable v1 API. Its methods accept and return
generated request and response models directly:

```python
from acp.experimental import v2

connection = v2.connect_to_agent(MyClient(), transport)
initialized = await connection.initialize(
    v2.schema.InitializeRequest(
        protocol_version=v2.PROTOCOL_VERSION,
        info=v2.schema.Implementation(name="my-client", version="1.0.0"),
    )
)
session = await connection.open_session(
    v2.schema.NewSessionRequest(cwd="/workspace")
)
```

`open_session()` returns an `ActiveSession`. A v2 prompt is accepted before the
agent finishes it, so consume session updates until `SessionStop` marks the
`running` to `idle` transition:

```python
await session.prompt(
    v2.schema.PromptRequest(
        session_id=session.session_id,
        prompt=[v2.schema.TextContentBlock(text="Hello")],
    )
)
stopped = await session.wait_for_idle()
```

## Negotiate v1 or v2

Use `ClientNegotiator` when the same client can speak both versions. It sends
exactly one `initialize` request and returns a version-tagged connection:

```python
from acp.experimental import (
    ClientNegotiator,
    NegotiatedV2,
    V1ClientConfig,
    V2ClientConfig,
)

negotiator = ClientNegotiator(
    transport,
    v1=V1ClientConfig(client=v1_client, initialize=v1_initialize),
    v2=V2ClientConfig(client=v2_client, initialize=v2_initialize),
)
negotiated = await negotiator.negotiate()

if isinstance(negotiated, NegotiatedV2):
    session = await negotiated.connection.open_session(v2_new_session)
else:
    session = await negotiated.connection.new_session(cwd="/workspace")
```

Agents that serve both versions use `AgentProtocolRouter`:

```python
from acp.experimental import AgentProtocolRouter

router = AgentProtocolRouter(v1=v1_agent, v2=v2_agent)
await router.run()
```

The selected runtime remains strict after initialization: v1 messages are not
accepted by a v2 connection, and v2 messages are not translated into v1 calls.
Only the initial v2 request is reduced to the common v1 initialization fields
when an agent selects v1.
