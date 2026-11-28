# Testing & Conformance

The SDK validates protocol behavior across four layers: local unit/contract tests, the official ACP Technology Compatibility Kit (TCK), cross-SDK wire interoperability, and real-world client/agent integration checks.

## Local test suite

Run the standard repository checks locally:

```bash
make gen-check   # Verify generated bindings match schema/
make check       # Lockfile consistency, Ruff lint/format, ty type check, deptry
make test        # Pytest suite (including doctests)
make docs-test   # Strict MkDocs build
```

The test suite under `tests/` covers:

- Generated schema models, lenient deserialization (`x-deserialize-*`), and golden wire fixtures (`tests/test_golden.py`, `tests/test_deserialize.py`, `tests/test_v2_schema.py`)
- JSON-RPC framing, cancellation (`$/cancel_request`), routing, and protocol negotiation (`tests/test_rpc.py`, `tests/test_protocol_negotiation.py`, `tests/test_v2_runtime.py`)
- Streamable HTTP and WebSocket client/server transports (`tests/http/`)
- Contrib helpers (`tests/contrib/`) and stdio subprocess lifecycles (`tests/real_user/`)

## ACP TCK conformance

Protocol-level compliance is verified against the official [ACP TCK](https://github.com/agentclientprotocol/acp-tck) using reference agents built strictly on public SDK APIs across four profiles:

- **v1 baseline (`acp.run_agent`)** — Stable v1 agent suite (`Agent` subclass and structural handler), expecting `CONFORMANT`.
- **v2 minimal (`v2.run_agent`, `AgentProtocolRouter(v2=...)`)** — Minimal capabilities, validating error codes (`-32022`, `-32600`, `-32601`, `-32602`) and disabled-capability handling with `CONFORMANT` and 0 failures.
- **v2 full capability (`v2.run_agent`, `AgentProtocolRouter(v2=...)`)** — All supported v2 capabilities enabled (sessions, fork/resume/replay, config options, elicitation, MCP-over-ACP, permissions, cancellation), expecting `CONFORMANT` and 0 failures.
- **Dual-stack negotiation (`AgentProtocolRouter(v1=..., v2=...)`)** — v1/v2 version selection (`ACP-INIT-208`), repeat `initialize`, and v2-to-v1 fallback with 0 failures on the initialization suite.

When modifying schema base classes or deserialization helpers (`src/acp/_schema_base.py`, `src/acp/_deserialize.py`), also verify v1 `model_validate` latency on hot models (`AgentMessageChunk`, `SessionNotification`) to keep v1 validation performance steady.

## Cross-SDK wire interoperability

For schema upgrades and client-side behavior ahead of TCK releases, run live bidirectional stdio checks pairing this SDK with the official [TypeScript SDK](https://github.com/agentclientprotocol/typescript-sdk) and [Rust SDK](https://github.com/agentclientprotocol/rust-sdk) across all four directions (Python client ↔ TS/Rust agent, TS/Rust client ↔ Python agent).

## Real-world client and agent verification

Real-world checks validate end-to-end behavior with ecosystem tools across two roles:

- **Real ACP clients** (exercising Python SDK agents): **Zed**, **T3 Code**, **Toad**, and **Lody**
- **Real ACP agents** (exercising Python SDK clients): **Gemini CLI** (`gemini --acp`), **Bub** (`bub-acp-server`), and **Mistral Vibe**

### Standard smoke playbook (`S1–S6`)

Use the same six scenarios across clients and agents when validating releases or transport changes:

1. **S1 · Handshake & negotiation** — `initialize` exchanges protocol version, info, and capabilities; unsupported versions return clean errors.
2. **S2 · Session & streaming** — `session/new` followed by `session/prompt` streams `agent_message_chunk` updates through turn completion.
3. **S3 · Tool calls & permissions** — Tool call progress/content/diff updates and `request_permission` approve/reject flows complete end to end.
4. **S4 · Cancellation & recovery** — Sending `session/cancel` or `$/cancel_request` stops active work and leaves the session usable for the next prompt.
5. **S5 · Session lifecycle & config** — Advertised capabilities (`session/list`, `load`/`resume`, `fork`, `configOptions`, `mcp-over-acp`) round-trip cleanly.
6. **S6 · Error handling & shutdown** — Invalid frames or unimplemented methods keep the connection healthy; closing exits cleanly with zero orphan processes.

### Running the opt-in Gemini CLI smoke test

When the Gemini CLI is installed and authenticated locally, run the built-in smoke test in `tests/test_gemini_example.py`:

```bash
ACP_ENABLE_GEMINI_TESTS=1 uv run python -m pytest tests/test_gemini_example.py
```

Optional environment variables:

- `ACP_GEMINI_BIN` — custom path to the `gemini` executable (defaults to `PATH` lookup)
- `ACP_GEMINI_TEST_ARGS` — extra flags forwarded to `examples/gemini.py`

You can also run the interactive bridge directly:

```bash
python examples/gemini.py --skip-trust --yolo
python examples/gemini.py --skip-trust --sandbox --model gemini-1.5-pro
```
