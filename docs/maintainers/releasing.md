# Release the SDK

The package release workflow and the documentation workflow run independently. Publishing a GitHub Release builds and uploads a package to PyPI. Documentation changes on `main` build and deploy the site without a package release.

## Prepare the release

Choose the upstream schema ref for each protocol version you are updating. The pinned refs live in `schema/VERSION` for v1 and `schema/v2/VERSION` for experimental v2.

For example, regenerate v1 bindings with a selected schema tag:

```bash
ACP_SCHEMA_VERSION=schema-v1.23.0 make gen-all
```

For v2, select the protocol explicitly:

```bash
uv run scripts/gen_all.py --protocol-version 2 --version schema-v2.0.0-alpha.5
```

These commands update the selected schema, generated models, method metadata, and handler signatures. Review their diff alongside the implementation and tests. To check generated schema and metadata against the cached inputs without rewriting files:

```bash
make gen-check
```

Choose a PEP 440 SDK version, update `pyproject.toml`, and run `uv lock` to keep the lockfile consistent. The SDK version and the upstream schema tag are separate identifiers.

Run the release checks:

```bash
make check
make test
make docs-test
```

Update documentation and examples for public API changes. When the CLI and credentials are available, run the opt-in Gemini smoke test with `ACP_ENABLE_GEMINI_TESTS=1 uv run python -m pytest tests/test_gemini_example.py`. Set `ACP_GEMINI_BIN` to select its executable and `ACP_GEMINI_TEST_ARGS` for additional CLI arguments.

## Review the changes

Use a Conventional Commit prefix such as `feat:`, `fix:`, or `docs:`. Call out schema regenerations in the PR description, including the selected refs, changed behaviors, and validation results. Link migration instructions for breaking API changes.

## Publish a GitHub Release

Once the changes land on `main`, create a tag and GitHub Release using the exact package version, such as `1.0.0rc3`. Use prerelease status for an alpha, beta, or release candidate.

Publishing the release triggers `.github/workflows/on-release-main.yml`. It checks out the release ref, substitutes the tag's version into `pyproject.toml`, builds the package with `uv build`, and uploads it with `uv publish` using the repository's `PYPI_TOKEN` secret.

Inspect the `set-version` and `publish` jobs and confirm the package appears on PyPI. A package publication does not deploy the documentation site. For site setup, deployment, and recovery, see [Deploy documentation](documentation.md).
