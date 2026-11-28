# Releasing

Releases track pinned upstream ACP schema tags (`schema/VERSION` for v1 and `schema/v2/VERSION` for experimental v2).

## Release checklist

1. Refresh generated bindings when advancing the upstream schema:
   ```bash
   ACP_SCHEMA_VERSION=schema-v1.35.0 make gen-all
   ACP_SCHEMA_VERSION=schema-v2.0.0-beta.4 make gen-v2
   ```
2. Bump the version in `pyproject.toml` (PEP 440, e.g. `1.0.0rc3` or `1.0.0`) and refresh `uv.lock` when dependencies change.
3. Run the verification gates:
   ```bash
   make gen-check && make check && make test && make docs-test
   ```
4. For schema, transport, or router releases, run the [ACP TCK and real-world smoke checks](testing.md) and summarize the results in the release PR.

## Publishing

Create and publish a GitHub Release on `main` whose tag matches the package version (for example `1.0.0`). `.github/workflows/on-release-main.yml` syncs the version, builds and publishes to PyPI, and deploys the MkDocs site to GitHub Pages.

To republish documentation for an existing tag, trigger **Actions → Publish release docs → Run workflow** on `main` with the target `release-tag`.
