# Migrating to ACP Python SDK 0.11

ACP Python SDK 0.11 updates the generated bindings to `schema-v1.16.0` and aligns the high-level interfaces with the new schema. Most applications only need to update method signatures and review the new unstable capabilities. Agents and clients that already pass keyword arguments are the easiest to migrate.

## 1. Regenerate schema-derived code

If your project vendors ACP schema files, generated models, or protocol metadata, regenerate them against the same upstream schema tag:

```bash
ACP_SCHEMA_VERSION=schema-v1.16.0 make gen-all
```

The SDK package version is `0.11.0`, while the protocol schema tag is `schema-v1.16.0`.

## 2. Update interface method signatures

Several generated request models changed field order or fields. The SDK now exposes those shapes through `Agent` and `Client` protocol methods. Prefer keyword calls when invoking connection helpers; keyword calls are stable across field-order changes.

### Client methods

Update client implementations from the 0.10 positional order:

```python
async def request_permission(self, options, session_id, tool_call, **kwargs): ...
async def write_text_file(self, content, path, session_id, **kwargs): ...
async def read_text_file(self, path, session_id, limit=None, line=None, **kwargs): ...
async def create_terminal(self, command, session_id, args=None, cwd=None, env=None, **kwargs): ...
```

to the 0.11 order:

```python
async def request_permission(self, session_id, tool_call, options, **kwargs): ...
async def write_text_file(self, session_id, path, content, **kwargs): ...
async def read_text_file(self, session_id, path, line=None, limit=None, **kwargs): ...
async def create_terminal(self, session_id, command, args=None, env=None, cwd=None, **kwargs): ...
```

### Agent methods

Update agent implementations from the 0.10 prompt and mode signatures:

```python
async def set_session_mode(self, mode_id, session_id, **kwargs): ...
async def prompt(self, prompt, session_id, message_id=None, **kwargs): ...
```

to the 0.11 signatures:

```python
async def set_session_mode(self, session_id, mode_id, **kwargs): ...
async def prompt(self, session_id, prompt, **kwargs): ...
```

The `message_id` field was removed from `PromptRequest`. If your client generated a user message ID before calling `conn.prompt(...)`, stop passing it there. Message IDs now belong to streamed content chunks such as `UserMessageChunk` and `AgentMessageChunk`.

## 3. Remove `session/model` handling

The generated `SetSessionModelRequest` and `SetSessionModelResponse` types are no longer exported, and the `Agent.set_session_model(...)` protocol method is gone. If your agent used this endpoint to switch models, move that behavior into session modes or configuration options exposed through `set_session_mode(...)` and `set_config_option(...)`.

## 4. Handle elicitation if your agent or client advertises it

0.11 adds schema and connection support for the unstable `elicitation/create` request and `elicitation/complete` notification.

Clients that advertise elicitation support should implement:

```python
from typing import Any

from acp import AcceptElicitationResponse, Client, CreateElicitationResponse, ElicitationMode


class MyClient(Client):
    async def create_elicitation(
        self,
        message: str,
        mode: ElicitationMode,
        **kwargs: Any,
    ) -> CreateElicitationResponse:
        return AcceptElicitationResponse(action="accept", content={})

    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        ...
```

Agents can request structured input through the connected client:

```python
from acp import (
    ElicitationFormSessionMode,
    ElicitationSchema,
    ElicitationStringPropertySchema,
)

response = await client_conn.create_elicitation(
    message="Choose a deployment target",
    mode=ElicitationFormSessionMode(
        session_id=session_id,
        requested_schema=ElicitationSchema(
            properties={"target": ElicitationStringPropertySchema(type="string")},
            required=["target"],
        ),
    ),
)
```

For URL-based flows, use `ElicitationUrlSessionMode` or `ElicitationUrlRequestMode` and call `complete_elicitation(...)` once the external flow finishes.

## 5. Review new session update variants

Clients that exhaustively match `session_update` variants should add the new plan update variants:

```python
from acp.schema import AgentPlanContentUpdate, AgentPlanRemovedUpdate

async def session_update(self, session_id, update, **kwargs):
    if isinstance(update, AgentPlanContentUpdate):
        ...
    elif isinstance(update, AgentPlanRemovedUpdate):
        ...
```

The existing full-plan `AgentPlanUpdate` variant remains available.

## 6. Review MCP server configuration types

Session creation, loading, forking, and resuming now accept `AcpMcpServer` in addition to HTTP, SSE, and stdio MCP server definitions:

```python
from acp.schema import AcpMcpServer, HttpMcpServer, McpServerStdio, SseMcpServer
```

If you validate `mcp_servers` with your own union, add `AcpMcpServer` to keep accepting all SDK-supported server types.

## 7. Re-run examples and checks

After changing signatures, run the standard gates:

```bash
make check
make test
```

Also run any example or integration that subclasses `Agent` or `Client`. Positional argument bugs usually surface there first; using keyword arguments for connection calls avoids most of them.
