# Run the ACP TCK locally

From the SDK root, run the full suite for `version=1` or `version=2`:

```sh
uv sync --locked --no-dev --python 3.14.5
mkdir -p build/tck
git clone https://github.com/agentclientprotocol/acp-tck build/acp-tck
git -C build/acp-tck checkout --detach 7c71c4d1af0994e73bdc100c9f7f6b98f2319cda
version=1
uv run --project build/acp-tck --locked --no-dev --python 3.14.5 acp-tck \
  --protocol-version "$version" --timeout 5 --cancel-prompt wait_for_cancel \
  --report-json "$PWD/build/tck/v$version.json" -- \
  "$PWD/.venv/bin/python" "$PWD/tests/tck/agent_v$version.py"
```

The CLI writes its native JSON report and returns its own exit code. CI uploads that report,
the log, and the exit code for each lane, including failures. Treat v2 as diagnostic: the
TCK's alpha.16 snapshot differs from the SDK's alpha.7 snapshot.
