# Releasing

Package releases track pinned upstream ACP schema tags from [`agentclientprotocol/agent-client-protocol`](https://github.com/agentclientprotocol/agent-client-protocol) (`schema/VERSION` for v1 and `schema/v2/VERSION` for experimental v2).

## Preparation checklist

1. **Refresh schema bindings** when targeting a new upstream schema tag:
   ```bash
   # v1 bindings
   ACP_SCHEMA_VERSION=schema-v1.35.0 make gen-all

   # Experimental v2 bindings
   ACP_SCHEMA_VERSION=schema-v2.0.0-beta.4 make gen-v2
   ```
2. **Bump the SDK version** in `pyproject.toml` using a PEP 440 version string (for example `1.0.0rc3` or `1.0.0`) and update `uv.lock` if dependencies changed.
3. **Run local verification gates:**
   ```bash
   make gen-check
   make check
   make test
   make docs-test
   ```
4. **Run conformance and ecosystem checks** for schema, transport, or router changes:
   - Run the [ACP TCK](testing.md#acp-tck-conformance) profiles (v1 baseline, v2 minimal/full capability, dual-stack router).
   - Run the [S1–S6 real-world smoke playbook](testing.md#real-world-client-and-agent-verification) against available ACP clients and agents, and include the summary in the release PR.
5. **Update docs and examples** whenever public APIs, schema models, or transports change.

## Commit and PR

- Use Conventional Commits (`release: 1.0.0`, `feat(schema)!: ...`, `docs: ...`).
- Record the targeted v1/v2 schema tags, verification outputs (`make check`, `make test`, TCK verdict), and user-visible highlights in the PR description.

## Publish via GitHub Release

Publishing is automated by `.github/workflows/on-release-main.yml`:

1. Create and publish a GitHub Release on `main` whose tag matches the package version (for example `1.0.0`).
2. The workflow syncs the version into `pyproject.toml`, builds the wheel/sdist, and publishes to PyPI via `uv publish`.
3. Once publishing succeeds, **Publish release docs** builds the MkDocs site from the release commit and deploys it to GitHub Pages.

## Republishing docs for an existing release

Open **Actions → Publish release docs → Run workflow**, select the `main` branch, and enter the published release tag in `release-tag`. The workflow rebuilds and deploys documentation for that tag to GitHub Pages.
