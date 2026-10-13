# Integration patterns

An integration maps an application's events, history, and user decisions to ACP handlers. The projects below use this SDK at that boundary. For a local stdio exchange, start with the [quickstart](../tutorials/quickstart.md) or [agent tutorial](../tutorials/agent.md).

## Adapt an existing agent runtime: Bub

The [Bub ACP adapter](https://github.com/bubbuild/bub-contrib/tree/5257e58fef85c735001995a6da0186c1b66403dc/packages/bub-acp-server) exposes Bub's runtime over ACP. Its [package metadata](https://github.com/bubbuild/bub-contrib/blob/5257e58fef85c735001995a6da0186c1b66403dc/packages/bub-acp-server/pyproject.toml) pins SDK `1.0.0rc2` and requires Python 3.12 or newer.

| Integration responsibility | Implementation to read |
| --- | --- |
| Map initialization, sessions, prompts, and cancellation to the runtime | `BubACPAgent` in [agent.py](https://github.com/bubbuild/bub-contrib/blob/5257e58fef85c735001995a6da0186c1b66403dc/packages/bub-acp-server/src/bub_acp_server/agent.py) |
| Convert model text and tool activity into `AgentMessageChunk`, `ToolCallStart`, and `ToolCallProgress` notifications | `ACPStreamRouter` in the same file |
| Select client-backed file and terminal tools from advertised capabilities | `ACPClientToolRuntime` and `build_client_tools` in [client_tools.py](https://github.com/bubbuild/bub-contrib/blob/5257e58fef85c735001995a6da0186c1b66403dc/packages/bub-acp-server/src/bub_acp_server/client_tools.py) |
| Retain session metadata and replay application history for `session/load` | Session methods in `BubACPAgent` |

The adapter calls SDK `run_agent` to serve stdio. Bub owns execution, tapes, and configuration; the SDK carries protocol messages and dispatches client callbacks. The adapter's [end-to-end test](https://github.com/bubbuild/bub-contrib/blob/5257e58fef85c735001995a6da0186c1b66403dc/packages/bub-acp-server/tests/test_e2e.py) shows client-backed file and terminal operations.

## Adapt a framework agent: PydanticAI

PydanticAI's [experimental ACP adapter](https://pydantic.dev/docs/ai/harness/acp/) wraps a PydanticAI agent. [`PydanticAIACPAgent`](https://github.com/pydantic/pydantic-ai/blob/0a28fc079668fa03040f0a3024f3bf1f1b63e819/src/pydantic_ai_harness/pydantic_ai_harness/experimental/acp/_adapter.py) implements the SDK's `Agent` interface, maps run events to session updates, and uses `request_permission` for deferred tool approvals. Its [server entry point](https://github.com/pydantic/pydantic-ai/blob/0a28fc079668fa03040f0a3024f3bf1f1b63e819/src/pydantic_ai_harness/pydantic_ai_harness/experimental/acp/_server.py) passes that adapter to `acp.run_agent`.

Session history and approval behavior stay with the adapter; protocol framing and callback dispatch come from the SDK. The [stdio test agent](https://github.com/pydantic/pydantic-ai/blob/0a28fc079668fa03040f0a3024f3bf1f1b63e819/tests/harness/experimental/acp/_demo_agent.py) uses PydanticAI's `TestModel` without a model service.

The adapter is experimental. Its [ACP extra](https://github.com/pydantic/pydantic-ai/blob/0a28fc079668fa03040f0a3024f3bf1f1b63e819/src/pydantic_ai_harness/pyproject.toml) requires SDK `>=0.11,<0.12`; use that dependency range when running this version of the adapter.

## Connect protocol events to notebook output

[`agent-client-kernel`](https://github.com/wiki3-ai/agent-client-kernel/blob/fc10043bf78d3e16af0117417aaf783472a194ef/agent_client_kernel/kernel.py) is an SDK-based client. Its `ACPClientImpl` handles session updates and permission requests. The kernel starts an agent subprocess, passes its pipes to `connect_to_agent`, initializes the connection, and sends cell input as a prompt. Client callbacks translate agent updates into notebook output.

The kernel owns the subprocess and notebook UI; the SDK dispatches updates and permission requests to its callbacks. Its [package metadata](https://github.com/wiki3-ai/agent-client-kernel/blob/fc10043bf78d3e16af0117417aaf783472a194ef/pyproject.toml) marks the project alpha and depends directly on the SDK Git repository. The [client tutorial](../tutorials/client.md) covers the same connection and callback structure in a standalone Python application.
