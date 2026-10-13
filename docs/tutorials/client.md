# Build a client

Implement `Client` callbacks in your application and use a connection helper to communicate with an agent. The client sends prompts and receives updates; it can also provide filesystem, terminal, and permission services that the agent calls.

## Start from a complete exchange

Follow the [quickstart](quickstart.md#make-the-calls-from-python) to create `hello_client.py` in the repository root. Run it from the checkout:

```bash
uv run --frozen python hello_client.py
```

Expected output:

```text
Hello ACP
Turn finished: end_turn
```

That example uses the checked-out SDK, starts the repository's echo agent, and shuts down the child process when the context manager exits. It needs no external agent installation, model account, or credentials. Keep it as a smoke check while developing your callbacks.

## Implement incoming callbacks

The quickstart's `EchoClient` subclasses `Client` and implements two methods:

| Callback | Your application's responsibility |
| --- | --- |
| `session_update(session_id, update, **kwargs)` | Route each event to the session's UI or state store; distinguish text, tool activity, and other update types |
| `request_permission(session_id, tool_call, options, **kwargs)` | Present the supplied options and return the selected option ID or cancellation |

Use generated models rather than hand-written response dictionaries. The [permission guide](../how-to/permissions.md) explains why selecting an option is not necessarily approval. The [session state guide](../how-to/session-state.md) shows how to keep interleaved sessions separate.

Your callback signatures receive snake-case fields. Request `_meta` entries are expanded into keyword arguments; collect those your handler does not use in `**kwargs`. Read the [client handler reference](../reference/api.md#client-handlers) for the complete types.

## Advertise the services you implement

The minimal client does not advertise filesystem or terminal services. Add `client_capabilities` to its `initialize` call when those handlers exist in your application. Filesystem capabilities select `read_text_file` and `write_text_file`; `terminal=True` requires the terminal lifecycle handlers.

Define how your application resolves workspace paths and enforces permissions before exposing those services. Protocol models validate message shape; they do not decide whether your application should allow an operation. Check the agent's advertised capabilities before calling optional methods such as loading a saved session.

## Own the connection lifetime

`spawn_agent_process` launches the agent and closes its connection and process when the async context exits. Use `connect_to_agent` when your application already owns byte streams or a message transport, and close the resulting connection during teardown.

The spawn helper inherits a limited environment, including `PATH`. Pass additional variables required by the agent through `env=...`. Its stderr is piped by default; `transport_kwargs={"stderr": None}` exposes diagnostics in your terminal.

Initialize the connection, create or load a session, and send prompts with that session ID. Response text arrives through updates; the v1 prompt response reports the stop reason. See [roles and session lifecycle](../explanation/lifecycle.md) for the flow, and [cancellation](../reference/protocol.md#request-cancellation) for stopping requests and prompt turns.

## Existing client code

[`examples/client.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/client.py) provides an interactive stdio client. Run it against the checked-out echo agent:

```bash
uv run --frozen python examples/client.py python examples/echo_agent.py
```

Type `Hello ACP`; it prints `| Agent: Hello ACP`. Its optional filesystem and terminal callbacks deliberately reject unsupported requests.

[`examples/gemini.py`](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/gemini.py) shows an interactive permission UI and filesystem callbacks. Its terminal methods are placeholders. Use it to study those implemented callbacks, rather than as a complete terminal service. The [integration patterns](../integrations/index.md) link another SDK client that connects protocol events to notebook output.
