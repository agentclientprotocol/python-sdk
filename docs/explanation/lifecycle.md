# Roles and session lifecycle

An ACP client starts or connects to an agent, creates sessions, sends prompts, and handles callbacks. The agent owns the work performed for each prompt. An editor is one kind of client; a Python application can be another.

`Agent` and `Client` are typed Protocol interfaces describing handler methods. You may subclass them or provide an object with the corresponding methods. The SDK's connections translate JSON-RPC requests into these handler calls and validate protocol models.

## A v1 exchange

```text
Client                                    Agent
  |--- initialize ------------------------->|
  |<-- protocol version and capabilities ---|
  |--- session/new ------------------------>|
  |<-- session ID --------------------------|
  |--- session/prompt ---------------------->|
  |<-- session/update (text, tools, ...) ----|
  |<-- more session/update notifications ----|
  |<-- prompt response with stop_reason -----|
```

The client sends its latest supported major protocol version. The agent returns that version if supported, or its latest supported version. A client that cannot use the selected version closes the connection. Omitted optional capabilities mean unsupported. These are application obligations in the v1 runtime; the connection helpers do not choose capabilities or reject a negotiated version for you. See the protocol's [initialization contract](https://agentclientprotocol.com/protocol/v1/initialization).

The client may need to authenticate before creating a session. A session has an ID and an absolute working directory; prompts and updates refer to that ID. Agents support text and resource links as baseline prompt content. Images, audio, and embedded resources depend on the agent's advertised prompt capabilities.

In v1, `prompt()` waits for the turn's response. Text and tool activity arrive separately through `session_update()`. Register callbacks before opening a session, and use the update's type to decide how to display it. The final `PromptResponse` reports why the turn stopped; it does not contain the streamed answer text.

A session can contain several prompt turns. Keeping a connection open does not automatically provide conversation storage or support for loading a previous session. Implement those behaviors in the agent and advertise the corresponding capabilities.

Updates can also arrive outside a prompt turn, for example when the agent advertises commands or replays a loaded session. Install update handlers before opening sessions and allow the first update for a session ID to create its state entry. See [prompt turns](https://agentclientprotocol.com/protocol/v1/prompt-turn) and [session setup](https://agentclientprotocol.com/protocol/v1/session-setup).

## Requests can travel in both directions

While processing a prompt, an agent can ask the client to read a file, run a terminal command, or request user approval. The client advertises the filesystem and terminal services it implements. Tool call notifications describe activity; permission requests ask for a decision. Those are separate messages.

The client and agent each need handlers for the methods their peer will call. Implement every capability your application advertises. The [API reference](../reference/api.md) gives the signatures, and the [permission guide](../how-to/permissions.md) shows how to interpret a decision.

## Cancel a turn

`await connection.cancel(session_id=session_id)` sends `session/cancel`, a notification with no response. Your agent's `cancel` handler stops the session's active work. The original `prompt` request still completes with `stop_reason="cancelled"`, after cleanup and any final updates. The client continues accepting updates until that response and resolves pending permission requests with a cancellation outcome. This differs from [cancelling an individual JSON-RPC request](../reference/protocol.md#request-cancellation).

## Process and transport lifetime

For stdio, the client normally starts an agent process and connects its stdin/stdout pipes. Agent diagnostics go to stderr. `spawn_agent_process` manages the connection and child-process shutdown inside an async context manager.

HTTP and WebSocket move protocol messages across a network transport while retaining the v1 connection API. Their [transport architecture](transports.md) explains how routing differs from stdio.

The [experimental v2 runtime](../reference/experimental/protocol-v2.md) has a different prompt contract: a response acknowledges the user message rather than waiting for processing to finish. Use its own models and handlers; the v1 flow above does not describe v2 completion.
