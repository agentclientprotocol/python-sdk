# Deploy documentation

The site documents `main`. Documentation deployment uses `.github/workflows/docs.yml` and runs independently of package releases.

## Build and preview locally

```bash
uv sync --frozen
make docs-test
make docs
```

`make docs-test` builds with MkDocs strict mode. `make docs` starts a local server at `http://127.0.0.1:8000/python-sdk/`. MkDocs renders the Markdown pages and API references from `src/acp/`. The `redirect_maps` setting in `mkdocs.yml` maps older page URLs to their current destinations and preserves URL fragments.

## GitHub Pages publishing

The repository publishes from the root of `gh-pages`. Settings → Pages uses **Deploy from a branch**, with `gh-pages` and `/ (root)` selected. The Documentation workflow builds with MkDocs strict mode and commits the generated site to that branch using `mkdocs gh-deploy`.

The workflow explicitly requests a Pages build after pushing. GitHub notes that pushes made with `GITHUB_TOKEN` do not trigger a Pages build automatically; see [publishing-source troubleshooting](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#troubleshooting-publishing-from-a-branch). The build request uses the [Pages API](https://docs.github.com/en/rest/pages/pages#request-a-github-pages-build).

The `github-pages` environment must allow the `gh-pages` branch used by GitHub's Pages deployment.

## Publish and recover

The Main workflow checks documentation on pull requests. The Documentation workflow runs when relevant files change on `main`, including documentation, source, examples, build configuration, and dependencies. It fetches the existing `gh-pages` history, builds strictly, and pushes a new site commit without a force push.

Only the main-only publication job receives `contents: write` to update `gh-pages` and `pages: write` to request its build. PR checks use read-only permissions. The Pages concurrency group serializes publication runs without cancelling an in-progress run.

To rebuild without another source commit, select **Documentation** in Actions and run it with `main` selected. Runs on other refs skip publication. This operation does not publish a Python package and does not require `PYPI_TOKEN` or a separate deployment token.

The build request queues publication asynchronously. Check both the Documentation run and GitHub's Pages build/deployment run to see when the [site](https://agentclientprotocol.github.io/python-sdk/) is deployed.
