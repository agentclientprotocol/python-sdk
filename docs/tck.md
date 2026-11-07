# Agent-side protocol checks with ACP TCK

The [ACP TCK](https://github.com/agentclientprotocol/acp-tck) is an independent stdio client. It
launches the probe agents in `tests/tck/` and checks their wire behavior. The probes use this
checkout's public SDK entry points and generated models; they do not implement JSON-RPC framing
or validation themselves.

| Lane | SDK entry point | TCK suite | SDK schema source |
| --- | --- | --- | --- |
| v1 | `acp.run_agent` | `--protocol-version 1` | `schema/VERSION` |
| v2 | `acp.experimental.v2.run_agent` | `--protocol-version 2` | `schema/v2/VERSION` |

The v1 lane uses TCK commit `b7f40fb3c003c45926547be5a1f7e2b374035815`, with a schema
snapshot from spec commit `beac4ba96190018e9f6d82e148ff275084c5aa3e`. The v2 lane uses
TCK commit `fa00ccdc13dfaa4b288a91877c1f3e468af4aa7d`, whose schema snapshot is
`2.0.0-alpha.5` from spec commit `9af0e9f748db9f4cc4c410a7b212ead4f98ae78c`.
The SDK's generated schemas come from `schema-v1.23.0` and `schema-v2.0.0-alpha.5`.
The v2 lane checks the same draft version, although TCK uses the upstream unstable superset
and the SDK uses the published schema. Reports record both TCK and SDK provenance.

## Run locally

Install the SDK in its own environment and check out both pinned TCK commits in sibling directories:

```sh
uv python install 3.14
uv sync --frozen --python 3.14
/usr/bin/gh repo clone agentclientprotocol/acp-tck ../acp-tck
git -C ../acp-tck checkout --detach b7f40fb3c003c45926547be5a1f7e2b374035815
git -C ../acp-tck worktree add --detach ../acp-tck-alpha5 fa00ccdc13dfaa4b288a91877c1f3e468af4aa7d
.venv/bin/python scripts/run_tck.py --tck-v1-dir ../acp-tck --tck-v2-dir ../acp-tck-alpha5
```

The runner verifies that the agent Python imports `acp` from this checkout. TCK has its own uv
environment and lockfile; it never supplies the agent's SDK package. It writes `build/tck/v1.json`,
`build/tck/v2.json`, `build/tck/manifest.json`, and a detailed log for each suite.

The v1 lane is a CI gate: the official TCK verdict and every mandatory test outcome must pass.
This catches cancellation tests that TCK's aggregate verdict would otherwise allow to skip.

The v2 lane gates mandatory outcomes only: any `MANDATORY` `FAIL`, `NOT_TESTED`, or unexpected
`SKIPPED` requirement fails CI. The two conditional auth checks `ACP-AUTH-206` and `ACP-AUTH-207`
are the sole exception — alpha.5 skips them when the agent advertises no auth methods, and the
runner accepts those skips only when the fixture's recorded capabilities report no `auth` and
each child test is `SKIPPED` on the pinned node id with the pinned skip reason exactly. There is
no substring matching: a missing or unparseable reason, a missing capabilities object, a `FAIL`
row, or any third skip stays gated. The v2 lane does not gate the TCK's own verdict (which
also folds in `CAPABILITY` failures) and never gates `ADVISORY` results, which stay as advisory
reporting.

The CI job uploads all reports and logs even when its gate fails.

The probes advertise only the capabilities they implement. Optional capability tests that remain
`SKIPPED` are outside the current probe coverage. The v1 probe intentionally leaves MCP connection
to application code, so its `ACP-SESSION-004` advisory result is not an SDK transport verdict.
Client-side behavior and HTTP/WebSocket transports are covered by the repository's own tests,
not by these TCK lanes.
