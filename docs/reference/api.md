# Core API

Import the main entry points from `acp` and generated payload models from `acp.schema`. This reference covers the default protocol v1 runtime. For draft protocol v2, use [its separate API](experimental/protocol-v2.md).

## Connection helpers

Use `run_agent` to serve an agent over stdio. Use `spawn_agent_process` when a client owns the agent subprocess, or `connect_to_agent` when it already has byte streams or a message transport.

For `run_agent`, `input_stream` is the writer for agent messages (stdout by default), and `output_stream` is the reader for client messages (stdin by default). Omit both for the standard stdio setup.

::: acp.core.run_agent
    options:
      show_docstring_parameters: false

::: acp.core.connect_to_agent

::: acp.stdio.spawn_agent_process

::: acp.stdio.spawn_client_process

## Agent handlers

`Agent` defines the callbacks implemented by your agent. The methods below cover the [agent tutorial](../tutorials/agent.md); additional capabilities require their corresponding handlers in `acp.interfaces.Agent`.

::: acp.interfaces.Agent
    options:
      members: [on_connect, initialize, new_session, prompt, cancel]
      show_if_no_docstring: true

## Client handlers

`Client` defines callbacks the agent can send to your application. Permission responses select a supplied option or cancel the request; updates are notifications. Implement filesystem and terminal methods before advertising those capabilities.

::: acp.interfaces.Client
    options:
      members: [on_connect, session_update, request_permission]
      show_if_no_docstring: true

## Calls on the returned connection

`spawn_agent_process` yields a connection and its process. `connect_to_agent` returns the same connection type. Use the helpers to create it; this section documents the methods you call after connecting.

::: acp.client.connection.ClientSideConnection
    options:
      members: [initialize, new_session, prompt, cancel, close]
      show_if_no_docstring: true

## Common payload models

Construct models with snake-case Python field names, such as `session_id` and `stop_reason`. Generated discriminator fields have defaults, so `TextContentBlock(text="Hello")` is enough to create a text block. Connection helpers handle wire serialization. If you serialize a model yourself, use `model_dump(by_alias=True)` for protocol field names.

::: acp.schema.TextContentBlock
    options:
      members: [type, text]
      inherited_members: [text]

::: acp.schema.AgentMessageChunk
    options:
      members: [session_update, content]
      inherited_members: [content]

::: acp.schema.PromptResponse
    options:
      members: [stop_reason]

::: acp.schema.RequestPermissionResponse
    options:
      members: [outcome]

See [protocol features and cancellation](protocol.md) for optional methods, and [contrib reference](experimental/contrib.md) for session state and tool call utilities.
