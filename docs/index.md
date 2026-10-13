# ACP Python SDK

Use this SDK to implement an ACP agent or client in Python, or to add ACP to an existing application. The SDK handles protocol models, JSON-RPC communication, and connection lifetime. Your code supplies the agent runtime, client callbacks, and session storage.

These docs follow `main`. For a released SDK, use the examples and source at its [release tag](https://github.com/agentclientprotocol/python-sdk/releases). Features on `main` can be newer than the default PyPI installation.

## Start with a working exchange

The [quickstart](tutorials/quickstart.md) runs a local echo agent and client, then shows the Python calls that initialize the connection, create a session, and send a prompt. It requires Python, Git, and uv. You can complete it without an editor or a model account.

To add the SDK to your own project:

```bash
pip install agent-client-protocol
# or:
uv add agent-client-protocol
```

## Choose your next task

| You want to | Read |
| --- | --- |
| Expose your agent through ACP | [Build an agent](tutorials/agent.md) |
| Implement client callbacks and manage a connection | [Build a client](tutorials/client.md) |
| Request and handle approval for a tool call | [Handle permissions](how-to/permissions.md) |
| Merge updates into a UI's session state | [Accumulate session state](how-to/session-state.md) |
| Map an existing runtime to ACP handlers | [Integration patterns](integrations/index.md) |

## Understand the API

Start with [roles and session lifecycle](explanation/lifecycle.md) to understand the flow of requests, updates, and prompt results. The [core API reference](reference/api.md) lists handler signatures, connection helpers, and commonly used models. [Protocol features and cancellation](reference/protocol.md) covers optional methods and capability checks.

## Experimental features

The default API uses protocol v1. [Protocol v2](reference/experimental/protocol-v2.md) has a separate runtime and different prompt semantics. [HTTP and WebSocket](reference/experimental/web-transports.md) are optional remote transports. [Contrib helpers](reference/experimental/contrib.md) provide tool call tracking and session state utilities. These APIs may change.

## Maintain or upgrade an integration

The migration guides cover [0.11](migration/0.11.md), [0.8](migration/0.8.md), and [0.7](migration/0.7.md). Maintainers can find [release steps](maintainers/releasing.md), [documentation deployment](maintainers/documentation.md), and [protocol route maintenance](maintainers/protocol-routes.md) in the navigation.

The [integration patterns](integrations/index.md) page links SDK calls to the application responsibilities in existing adapters. Report bugs in [GitHub Issues](https://github.com/agentclientprotocol/python-sdk/issues) or discuss integration questions in [GitHub Discussions](https://github.com/agentclientprotocol/python-sdk/discussions).
