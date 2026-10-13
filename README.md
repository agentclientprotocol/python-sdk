<a href="https://agentclientprotocol.com/">
  <img alt="Agent Client Protocol" src="https://zed.dev/img/acp/banner-dark.webp">
</a>

# Agent Client Protocol SDK for Python

Implement ACP agent and client integrations in Python. The SDK provides typed protocol models, asynchronous handler interfaces, and connection helpers. Your application supplies the agent runtime, client callbacks, and session state.

Requires Python 3.10 through 3.14.

## Install

```bash
pip install agent-client-protocol
```

For a project managed with uv:

```bash
uv add agent-client-protocol
```

Import the package as `acp`:

```python
from acp.schema import TextContentBlock

message = TextContentBlock(text="Hello ACP")
```

## Run an agent and client

The example client starts the echo agent, sends your message, and prints the streamed reply. Install Git and uv, then run:

```bash
git clone https://github.com/agentclientprotocol/python-sdk.git
cd python-sdk
uv sync --frozen
uv run --frozen python examples/client.py python examples/echo_agent.py
```

Type `Hello ACP` at the prompt. The client starts the agent and prints `| Agent: Hello ACP`. See the [quickstart](https://agentclientprotocol.github.io/python-sdk/tutorials/quickstart/) for the initialization and session calls behind this exchange.

This runs the checked-out SDK and examples. Installing the package alone does not create an `examples/` directory. The documentation follows `main`, which can include features beyond the latest stable PyPI release.

## Build your integration

| Task | Guide |
| --- | --- |
| Implement an agent | [Build an agent](https://agentclientprotocol.github.io/python-sdk/tutorials/agent/) |
| Implement client callbacks and manage an agent process | [Build a client](https://agentclientprotocol.github.io/python-sdk/tutorials/client/) |
| Handle tool calls and permissions | [Permission guide](https://agentclientprotocol.github.io/python-sdk/how-to/permissions/) |
| Look up signatures and models | [Core API reference](https://agentclientprotocol.github.io/python-sdk/reference/api/) |
| Study adapters built with this SDK | [Integration patterns](https://agentclientprotocol.github.io/python-sdk/integrations/) |

Protocol v2, remote transports, and `acp.contrib` are experimental. Check their documentation before depending on their APIs.

## Contribute

```bash
make install
make check
make test
make docs-test
```

Report bugs in [GitHub Issues](https://github.com/agentclientprotocol/python-sdk/issues). Include the SDK version, the agent or client you used, and a minimal reproduction. See the [release guide](https://agentclientprotocol.github.io/python-sdk/maintainers/releasing/) for schema regeneration and package publishing.
