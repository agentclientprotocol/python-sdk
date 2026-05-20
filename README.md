<a href="https://agentclientprotocol.com/">
  <img alt="Agent Client Protocol" src="https://zed.dev/img/acp/banner-dark.webp">
</a>

# Agent Client Protocol (Python)

[![PyPI](https://img.shields.io/pypi/v/agent-client-protocol?style=flat-square)](https://pypi.org/project/agent-client-protocol/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-client-protocol?style=flat-square)](https://pypi.org/project/agent-client-protocol/)
[![License](https://img.shields.io/github/license/agentclientprotocol/python-sdk?style=flat-square)](https://github.com/agentclientprotocol/python-sdk/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue?style=flat-square)](https://agentclientprotocol.github.io/python-sdk/)

Build ACP-compliant agents and clients in Python with typed schema models, asyncio transports, helper builders, and runnable examples.

> Releases track the upstream ACP schema, so payloads stay aligned with the current spec.

## Table of Contents

- [Why this SDK](#why-this-sdk)
- [What you get](#what-you-get)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Who it is for](#who-it-is-for)
- [Examples](#examples)
- [Project layout](#project-layout)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Why this SDK

If you want to build or embed ACP-compatible software in Python, this repository gives you the repetitive pieces out of the box:

- generated ACP schema models
- stdio JSON-RPC transport plumbing
- async agent and client base classes
- helper builders for content, tool calls, and updates
- example apps you can run or adapt

## What you get

- **Spec parity**: `acp.schema` tracks ACP releases with generated Pydantic models.
- **Runtime ergonomics**: Async lifecycle helpers keep custom agents small and readable.
- **Composable helpers**: `acp.helpers` mirrors the Go and TypeScript SDK ergonomics.
- **Useful contrib modules**: Permission brokers, session accumulators, and tool call trackers reflect real deployment patterns.
- **Runnable demos**: Examples cover streaming, permissions, Gemini bridging, and duet-style integrations.

## Installation

```bash
pip install agent-client-protocol
# or
uv add agent-client-protocol
```

Python `3.10` through `3.14` are supported.

## Quickstart

### 1. Install the package

```bash
pip install agent-client-protocol
```

### 2. Explore the examples

```bash
git clone https://github.com/agentclientprotocol/python-sdk.git
cd python-sdk
uv sync
uv run python examples/echo_agent.py
```

### 3. Read the docs

Start with the [Quickstart guide](https://agentclientprotocol.github.io/python-sdk/quickstart/) for editor wiring, echo-agent validation, and launch recipes.

## Who it is for

- **Agent authors** building ACP-compatible assistants in Python
- **Client integrators** embedding ACP parties into existing apps or CLIs
- **Tooling teams** experimenting with streaming UX, permission flows, and transport abstractions

See the [Use Cases list](https://agentclientprotocol.github.io/python-sdk/use-cases/) for concrete adopters and integration examples.

## Examples

The `examples/` directory includes progressively richer demos, including:

- echo-style starter agents
- streaming integrations
- permission workflows
- Gemini bridge experiments
- duet-style interaction patterns

## Project layout

- `src/`: package source code
- `schema/`: generated ACP schema inputs
- `examples/`: runnable demos and integration samples
- `docs/`: documentation source
- `tests/`: automated test coverage

## Documentation

- [Docs home](https://agentclientprotocol.github.io/python-sdk/)
- [Quickstart](https://agentclientprotocol.github.io/python-sdk/quickstart/)
- [Use cases](https://agentclientprotocol.github.io/python-sdk/use-cases/)
- [Examples directory](https://github.com/agentclientprotocol/python-sdk/tree/main/examples)

## Contributing

Contributions that improve coverage, tooling, docs clarity, or examples are welcome. Please check [CONTRIBUTING.md](CONTRIBUTING.md) and the development configuration in `pyproject.toml`, `tox.ini`, and `.pre-commit-config.yaml` before opening a PR.

## License

See [LICENSE](LICENSE).
