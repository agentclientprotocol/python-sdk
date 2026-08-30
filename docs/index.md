<a href="https://agentclientprotocol.com/">
  <img alt="Agent Client Protocol" src="https://zed.dev/img/acp/banner-dark.webp">
</a>

# Agent Client Protocol SDK (Python)

Ship ACP-compatible agents and clients in Python without rebuilding JSON-RPC transports or schema models. This SDK mirrors each ACP release so your integrations stay interoperable with editors, CLIs, and hosted clients.

## Install & verify

```bash
pip install agent-client-protocol
# or
uv add agent-client-protocol
```

Next steps live in the [Quickstart](quickstart.md): launch the echo agent, wire it to Zed (or another ACP client), and exercise the programmatic spawn helpers.

## ACP at a glance

- ACP is the stdio protocol that lets “clients” (editors, shells, CLIs) orchestrate AI “agents.”
- Sessions exchange structured payloads (`session/update`, permission prompts, tool calls) defined in the upstream schema.
- Matching the schema version keeps your Python integrations compatible with tools such as Zed, Gemini CLI, or kimi-cli.

## SDK building blocks

- `acp.schema`: generated Pydantic models with defaults for protocol discriminator fields.
- `acp.agent` / `acp.client`: async base classes, JSON-RPC supervision, and lifecycle orchestration.
- `acp.contrib`: experimental utilities (session accumulators, permission brokers, tool call trackers) harvested from production deployments.
- `examples/`: runnable agents, clients, duet demos, and the Gemini CLI bridge.

## Quick links

| Need | Link |
| --- | --- |
| Quickstart walkthrough | [quickstart.md](quickstart.md) |
| Real-world adopters | [use-cases.md](use-cases.md) |
| Contrib helpers | [contrib.md](contrib.md) |
| Releasing workflow | [releasing.md](releasing.md) |
| Upgrade to 0.11 | [migration-guide-0.11.md](migration-guide-0.11.md) |
| Example scripts | [github.com/agentclientprotocol/python-sdk/tree/main/examples](https://github.com/agentclientprotocol/python-sdk/tree/main/examples) |

## Choose a path

- **Just exploring?** Skim [use-cases.md](use-cases.md) to see how kimi-cli, agent-client-kernel, and others use the SDK.
- **Building agents?** Copy `examples/echo_agent.py` or `examples/agent.py`, then use the generated models directly. Use `acp.contrib` for permission workflows and other stateful patterns.
- **Embedding clients?** Start with `examples/client.py` or the `spawn_agent_process` / `spawn_client_process` helpers in the [Quickstart](quickstart.md#programmatic-launch).

## Reference material

- [Quickstart](quickstart.md) — installation, editor wiring, and programmatic launch walkthroughs.
- [Use Cases](use-cases.md) — real adopters with succinct descriptions of what they build.
- [Experimental Contrib](contrib.md) — deep dives on the `acp.contrib` utilities.
- [Releasing](releasing.md) — schema upgrade process, versioning policy, and publishing checklist.
- [0.11 Migration Guide](migration-guide-0.11.md) — interface signature updates, elicitation, and schema notes for 0.10 users.

Need API-level details? Browse the source in `src/acp/` or generate docs with `mkdocstrings`.

## Feedback & support

- Open issues or discussions on GitHub for bugs, feature requests, or integration help.
- Join [GitHub Discussions](https://github.com/agentclientprotocol/python-sdk/discussions) to swap ideas.
- Chat with the community on [agentclientprotocol.zulipchat.com](https://agentclientprotocol.zulipchat.com/).
- Follow ACP roadmap updates at [agentclientprotocol.com](https://agentclientprotocol.com/).
