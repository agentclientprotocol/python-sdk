<a href="https://agentclientprotocol.com/">
  <img alt="Agent Client Protocol" src="https://zed.dev/img/acp/banner-dark.webp">
</a>

# Agent Client Protocol SDK (Python)

Build ACP agents and clients in Python with generated Pydantic models and asyncio transports over stdio, Streamable HTTP, and WebSocket.

```bash
pip install agent-client-protocol
# Optional: pip install "agent-client-protocol[http]"
```

## Where to start

- **[Quickstart](quickstart.md)** — Run the echo agent, wire it into an editor or CLI client, and write your first custom agent.
- **[Protocol Features (v1)](protocol-features.md)** — Use unstable opt-in features, session notices, stateless MCP-over-ACP, request cancellation, and extension methods.
- **[Web Transport](web-transport.md)** — Run ACP over Streamable HTTP (HTTP/2 + SSE) and WebSocket with Starlette.
- **[Experimental Protocol v2](experimental-v2.md)** — Explore the preview v2 API (`schema-v2.0.0-beta.4`) and dual-stack `AgentProtocolRouter`.
- **[Contrib Utilities](contrib.md)** — Reuse stateful helpers for session aggregation, tool call tracking, and permission prompts.

## Ecosystem and operations

- **[Use Cases](use-cases.md)** — Agents and clients built with this SDK.
- **[Testing & Conformance](testing.md)** — Local checks, ACP TCK conformance, and real-world interoperability checks.
- **[Releasing](releasing.md)** — Schema updates and release workflow.

Questions and feedback are welcome in [GitHub Discussions](https://github.com/agentclientprotocol/python-sdk/discussions) and on [Zulip](https://agentclientprotocol.zulipchat.com/).
