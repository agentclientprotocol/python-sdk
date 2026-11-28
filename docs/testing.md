# Testing & Conformance

We validate SDK changes at three levels: fast local checks, official ACP TCK conformance, and real-world interoperability.

## Local checks

```bash
make gen-check   # Verify generated bindings match schema/
make check       # Lockfile, Ruff lint/format, ty type check, deptry
make test        # Pytest suite (including doctests)
make docs-test   # Strict MkDocs build
```

## Protocol conformance (ACP TCK)

Schema, router, and transport changes are verified against the official [ACP TCK](https://github.com/agentclientprotocol/acp-tck) using reference agents built strictly on public SDK APIs:

- **v1 baseline**: `acp.run_agent` passes the v1 suite (`CONFORMANT`).
- **v2 minimal and full capability**: both `v2.run_agent` and `AgentProtocolRouter(v2=...)` pass with 0 failures (`CONFORMANT`).
- **Dual-stack negotiation**: `AgentProtocolRouter(v1=..., v2=...)` passes v1/v2 version selection, repeat `initialize`, and v2-to-v1 fallback.

When changing shared schema or deserialization code, also check v1 `model_validate` latency on hot models (`AgentMessageChunk`, `SessionNotification`) to avoid regressions.

## Real-world interoperability

Before major releases or schema bumps, we run live end-to-end checks across two directions:

- **Real ACP clients** driving Python SDK agents: **Zed**, **T3 Code**, **Toad**, and **Lody**.
- **Real ACP agents** driven by Python SDK clients: **Gemini CLI** (`gemini --acp`), **Bub** (`bub-acp-server`), and **Mistral Vibe**, plus live stdio checks against the official [TypeScript](https://github.com/agentclientprotocol/typescript-sdk) and [Rust](https://github.com/agentclientprotocol/rust-sdk) SDKs.

A standard end-to-end pass exercises one continuous flow:

1. **Handshake** — `initialize` negotiates version and capabilities cleanly.
2. **Streaming turn** — `session/new` and `session/prompt` stream updates through turn completion.
3. **Tools & permissions** — tool call updates and `request_permission` approve/reject flows complete cleanly.
4. **Cancel & continue** — cancelling mid-turn stops active work and leaves the session ready for the next prompt.
5. **Session lifecycle & exit** — advertised session/config/MCP features work and closing leaves zero orphan processes.

### Opt-in Gemini CLI smoke test

With the Gemini CLI installed and authenticated locally:

```bash
ACP_ENABLE_GEMINI_TESTS=1 uv run python -m pytest tests/test_gemini_example.py
```

Set `ACP_GEMINI_BIN` to override the executable path or `ACP_GEMINI_TEST_ARGS` to pass extra flags to `examples/gemini.py`.
