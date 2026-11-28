# Contrib Utilities

> **Experimental.** Modules under `acp.contrib` package opt-in stateful helpers for common client and agent workflows. Their APIs may evolve across releases.

## Available helpers

- **`acp.contrib.session_state.SessionAccumulator`** — aggregates incoming `SessionNotification` updates into immutable `SessionSnapshot` models for UI rendering.
- **`acp.contrib.tool_calls.ToolCallTracker`** — tracks active tool call lifecycles and emits canonical `ToolCallStart` / `ToolCallProgress` updates.
- **`acp.contrib.permissions.PermissionBroker`** — coordinates `request_permission` prompts using tracked tool call state and default approve/reject options.

## `SessionAccumulator` (`acp.contrib.session_state`)

Feed raw `SessionNotification` events into one accumulator per UI controller and render from immutable snapshots:

- `apply(notification)` merges message chunks, plan updates, mode/config changes, and `tool_call` / `tool_call_update` payloads (including out-of-order updates).
- `snapshot()` returns an immutable `SessionSnapshot` Pydantic model with copied tool input/output payloads.
- `subscribe(callback)` pushes updated snapshots to subscribers whenever state changes.
- `auto_reset_on_session_change` resets accumulated state automatically when a new `session_id` arrives.

## `ToolCallTracker` and `PermissionBroker` (`acp.contrib.tool_calls`, `acp.contrib.permissions`)

Share a single `ToolCallTracker` with `PermissionBroker` in your agent loop to keep tool updates and permission requests synchronized:

- `ToolCallTracker.start()`, `progress()`, and `append_stream_text()` emit `ToolCallStart` and `ToolCallProgress` updates while maintaining an in-memory view via `view()` and `tool_call_model()`.
- `PermissionBroker.request_for()` builds and sends `request_permission` calls using tracked tool call state (or an explicit `ToolCall`) and optional extra content blocks.
- `default_permission_options()` provides the standard **Approve**, **Approve for session**, and **Reject** option triple. Omitting `options` or passing `None` applies the broker's defaults; passing an empty list raises `MissingPermissionOptionsError` before sending the request.
