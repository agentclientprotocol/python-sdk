<a href="https://agentclientprotocol.com/">
  <img alt="Agent Client Protocol" src="https://zed.dev/img/acp/banner-dark.webp">
</a>

# Agent Client Protocol SDK (Python)

Build ACP-compliant agents and clients in Python using generated Pydantic v2 models, asyncio JSON-RPC transports, and typed protocol interfaces.

## Installation

```bash
pip install agent-client-protocol
# or
uv add agent-client-protocol
```

Optional extras:

```bash
pip install "agent-client-protocol[http]"     # Streamable HTTP + WebSocket
pip install "agent-client-protocol[logfire]"  # Logfire / OpenTelemetry
```

## Modules

- **`acp` and `acp.schema`** — Stable ACP v1 runtime (`run_agent`, `connect_to_agent`, `spawn_agent_process`, `Agent`, `Client`) and generated Pydantic models (`schema-v1.35.0`).
- **`acp.experimental.v2`** — Preview ACP v2 runtime and generated models (`schema-v2.0.0-beta.4`), plus `AgentProtocolRouter` for dual-stack negotiation.
- **`acp.http` and `acp.ws`** — Experimental Streamable HTTP (SSE + HTTP/2) and WebSocket client/server transports (`[http]` extra).
- **`acp.contrib`** — Experimental stateful helpers: `SessionAccumulator`, `ToolCallTracker`, and `PermissionBroker`.

## Guides

- **[Quickstart](quickstart.md)** — Install the SDK, run the echo agent, connect from an ACP client, and launch processes programmatically.
- **[Protocol Features (v1)](protocol-features.md)** — Session management, authentication, providers, notices, elicitation, stateless MCP-over-ACP, request cancellation, and extension methods.
- **[Web Transport (HTTP & WebSocket)](web-transport.md)** — Serve and consume ACP over Streamable HTTP and WebSocket with Starlette and HTTP/2.
- **[Experimental Protocol v2](experimental-v2.md)** — Preview v2 API, expanded handlers, patch updates, replay cursors, and dual-stack routing.
- **[Contrib Utilities](contrib.md)** — Higher-level session state accumulators, tool call trackers, and permission brokers.
- **[Use Cases](use-cases.md)** — Production agents and clients built with the SDK.
- **[Testing & Conformance](testing.md)** — Local test suites, ACP TCK conformance profiles, cross-SDK interop, and real-world client/agent verification.
- **[Releasing](releasing.md)** — Schema upgrades, verification gates, and PyPI/docs release workflow.

## Community

- Issues and pull requests: [agentclientprotocol/python-sdk](https://github.com/agentclientprotocol/python-sdk)
- Discussions: [GitHub Discussions](https://github.com/agentclientprotocol/python-sdk/discussions)
- Community chat: [agentclientprotocol.zulipchat.com](https://agentclientprotocol.zulipchat.com/)
- Protocol specification: [agentclientprotocol.com](https://agentclientprotocol.com/)
