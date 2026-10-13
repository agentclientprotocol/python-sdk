# Experimental contrib helpers

`acp.contrib` provides optional helpers for session state, tool call tracking, and permission requests. These APIs are experimental and may change. For worked examples, see [Handle permissions](../../how-to/permissions.md) and [Accumulate session state](../../how-to/session-state.md).

- `session_state.SessionAccumulator` merges supported updates into a session snapshot.
- `tool_calls.ToolCallTracker` and `permissions.PermissionBroker` associate tool call updates with permission requests.

## SessionAccumulator (`acp.contrib.session_state`)

Use it when your UI needs to merge tool call updates and retain message chunks from `SessionNotification` events.

**Key capabilities**

- `SessionAccumulator.apply(notification)` applies `tool_call` and `tool_call_update` payloads. Progress without a start creates a provisional entry; a later start can replace its fields.
- `snapshot()` returns a `SessionSnapshot` containing full plan updates, the current mode, commands, and message chunks. The outer model is frozen; nested values remain mutable.
- `subscribe(callback)` lets you push snapshots into stores or UI components whenever state changes.
- Automatic reset on session-change (toggle with `auto_reset_on_session_change`).

Create one accumulator per session. It tracks tool call starts/progress, full `AgentPlanUpdate` events, current mode, available commands, and user/agent/thought chunks. Handle session metadata, config options, usage, notices, compaction, incremental plan events, and tool names separately if your UI needs them.

## ToolCallTracker & PermissionBroker (`acp.contrib.tool_calls`, `acp.contrib.permissions`)

Use them when your agent needs to associate streamed tool call updates and permission requests with the same ID.

- `ToolCallTracker.start()/progress()/append_stream_text()` emits canonical `ToolCallStart` / `ToolCallProgress` updates and keeps an in-memory view via `view()` or `tool_call_model()`.
- `PermissionBroker.request_for()` wraps `requestPermission` RPCs. It reuses tracker state (or a provided `ToolCall`), lets you append extra content, and defaults to Approve / Approve for session / Reject options.
- Omit `options` or pass `None` to use broker defaults. An explicit empty option list raises `MissingPermissionOptionsError` before the permission request is sent; the same applies to empty `default_options` when no per-request options are supplied.
- `default_permission_options()` exposes that canonical option triple if you need to customise it.

Keep a tracker with the session. Its methods return models; your agent sends those models through `session_update`. Share the tracker with `PermissionBroker` to include the latest tool call state in a request.

## API boundaries

The contrib layer uses the generated models in `acp.schema`:

- Schema data classes continue to live in `acp.schema`. Contrib helpers clone them with `.model_copy(deep=True)` before mutation.
- The core runtime does not import contrib modules implicitly; your application opts in.
- Your application enforces permission policy and performs tool operations. The broker requests a decision; it does not execute a tool.

Report helper bugs or API feedback in [GitHub Issues](https://github.com/agentclientprotocol/python-sdk/issues).
