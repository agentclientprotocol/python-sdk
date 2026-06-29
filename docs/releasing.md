# Releasing

Every package release tracks an upstream ACP schema tag from [`agentclientprotocol/agent-client-protocol`](https://github.com/agentclientprotocol/agent-client-protocol). Follow this checklist to stay in lockstep.

## Prep checklist

1. **Choose the schema tag** (e.g. `schema-v1.16.0`) and regenerate artifacts:
   ```bash
   ACP_SCHEMA_VERSION=schema-v1.16.0 make gen-all
   ```
   This refreshes `schema/` and the generated `src/acp/schema.py`.
2. **Bump the SDK version** in `pyproject.toml` using a PEP 440 version string (for example `0.9.0a1` for an alpha release), and sync `uv.lock` if the lockfile is tracked.
3. **Run the standard gates:**
   ```bash
   make check   # Ruff format/lint, type analysis, dep hygiene
   make test    # pytest + doctests
   ```
4. **Refresh docs + examples** so user-facing flows (e.g. Gemini bridge) reflect behaviour in the new schema.

## Commit & review

- Keep the diff tight: regenerated schema files, version bumps, doc updates, and any required fixture refresh (goldens, RPC tests, etc.).
- Use a Conventional Commit such as `release: 0.9.0a1`.
- In the PR description, capture:
  - The ACP schema tag you targeted.
  - Output from `make check` / `make test` (and optional Gemini tests if you ran them).
  - Behavioural or API highlights that reviewers should focus on.

## Publish via GitHub Release

Releases are automated by `on-release-main.yml` once the PR lands on `main`.

1. Draft a GitHub Release for the new tag (the UI creates the tag if missing).
   Use the exact package version as the tag, for example `0.9.0a1` or `0.9.0`.
2. Publishing the release triggers the workflow, which:
   - Syncs the tag back into `pyproject.toml`.
   - Builds and uploads to PyPI via `uv publish` using `PYPI_TOKEN`.
   - Deploys updated docs with `mkdocs gh-deploy`.

No local build/publish steps are needed—just provide a clear release summary (highlights, compatibility notes, migration tips).

## Extra tips

- Breaking schema bumps often mean updating `tests/test_golden.py`, `tests/test_rpc.py`, and any examples touched by new fields.
- Use `make clean` if you need a fresh slate before re-running `make gen-all`.
- When available, run the Gemini smoke test (`ACP_ENABLE_GEMINI_TESTS=1`, set `ACP_GEMINI_BIN`) to catch regressions early.
