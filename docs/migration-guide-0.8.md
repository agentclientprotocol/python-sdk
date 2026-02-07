# Migrating to ACP Python SDK 0.8

ACP 0.8 keeps the 0.7 public surface but aligns the SDK with the latest ACP schema and tightens a few runtime behaviors. Most teams only need to review the updated schema and terminal helpers. This guide calls out the changes that can affect downstream agents, clients, and tests.

## 1. ACP schema bumped to 0.10.8

- Regenerate any internal copies of ACP schema-derived artifacts against 0.10.8.
- If you vendor schema types, run `make gen-all` or your equivalent pipeline.
- Helper types now include `SessionInfoUpdate` in the `SessionUpdate` union, so downstream code that exhaustively matches update variants should include it.

## 2. `TerminalHandle` removal

`TerminalHandle` is no longer part of the public API. If you referenced it directly, switch to the request/response models and terminal IDs returned by `CreateTerminalRequest`/`CreateTerminalResponse`.

Typical adjustment:

```python
# Before (0.7.x)
handle = await conn.create_terminal(...)
await conn.terminal_output(session_id=..., terminal_id=handle.id)

# After (0.8.x)
resp = await conn.create_terminal(...)
await conn.terminal_output(session_id=..., terminal_id=resp.terminal_id)
```

## 3. Larger default stdio buffer limits

The default stdio reader limit is now 50MB to support multimodal payloads. If you run in memory-constrained environments, explicitly set `stdio_buffer_limit_bytes` when calling `run_agent`.

```python
await run_agent(agent, stdio_buffer_limit_bytes=2 * 1024 * 1024)
```

## 4. Documentation and quickstart updates

Docs and settings examples have been refreshed for ACP 0.10.8. If you maintain internal onboarding material, sync it with the latest docs in `docs/`.
