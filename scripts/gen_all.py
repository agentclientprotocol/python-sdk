#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts import gen_meta, gen_schema, gen_signature  # noqa: E402  pylint: disable=wrong-import-position

SCHEMA_DIR = ROOT / "schema"
SCHEMA_JSON = SCHEMA_DIR / "schema.json"
META_JSON = SCHEMA_DIR / "meta.json"
VERSION_FILE = SCHEMA_DIR / "VERSION"

DEFAULT_REPO = "agentclientprotocol/agent-client-protocol"
LEGACY_SCHEMA_PATHS = ("schema/schema.unstable.json", "schema/meta.unstable.json")
V1_SCHEMA_PATHS = ("schema/v1/schema.unstable.json", "schema/v1/meta.unstable.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate schema.py and meta.py from the ACP schema.")
    parser.add_argument(
        "--version",
        "-v",
        help=(
            "Git ref (tag/branch) of agentclientprotocol/agent-client-protocol to fetch the schema from. "
            "If omitted, uses the cached schema files or falls back to 'main' when missing."
        ),
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("ACP_SCHEMA_REPO", DEFAULT_REPO),
        help="Source repository providing schema.json/meta.json (default: %(default)s)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading schema files even when a version is provided.",
    )
    parser.set_defaults(format_output=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force schema download even if the requested ref is already cached locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    version = args.version or os.environ.get("ACP_SCHEMA_VERSION")
    repo = args.repo
    should_download = _should_download(args, version)

    if should_download:
        ref = resolve_ref(version)
        download_schema(repo, ref)
    else:
        ref = resolve_ref(version) if version else _cached_ref()

    if not (SCHEMA_JSON.exists() and META_JSON.exists()):
        print("schema/schema.json or schema/meta.json missing; run with --version to fetch them.", file=sys.stderr)
        sys.exit(1)

    gen_schema.generate_schema()
    gen_meta.generate_meta()
    gen_signature.gen_signature(ROOT / "src" / "acp")

    if ref:
        print(f"Generated schema using ref: {ref}")
    else:
        print("Generated schema using local schema files")


def _should_download(args: argparse.Namespace, version: str | None) -> bool:
    env_override = os.environ.get("ACP_SCHEMA_DOWNLOAD")
    if env_override is not None:
        return env_override.lower() in {"1", "true", "yes"}
    if args.no_download:
        return False
    if version:
        if not SCHEMA_JSON.exists() or not META_JSON.exists():
            return True
        cached = _cached_ref()
        if args.force:
            return True
        return cached != resolve_ref(version)
    return not (SCHEMA_JSON.exists() and META_JSON.exists())


def resolve_ref(version: str | None) -> str:
    if not version:
        return "refs/heads/main"
    if version.startswith("refs/"):
        return version
    if re.fullmatch(r"schema-v\d+\.\d+\.\d+", version):
        return f"refs/tags/{version}"
    if re.fullmatch(r"v?\d+\.\d+\.\d+", version):
        value = version if version.startswith("v") else f"v{version}"
        return f"refs/tags/{value}"
    return f"refs/heads/{version}"


def download_schema(repo: str, ref: str) -> None:
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        schema_data, meta_data = fetch_schema_pair(repo, ref)
    except RuntimeError as exc:  # pragma: no cover - network error path
        print(exc, file=sys.stderr)
        sys.exit(1)

    SCHEMA_JSON.write_text(json.dumps(schema_data, indent=2), encoding="utf-8")
    META_JSON.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
    VERSION_FILE.write_text(ref + "\n", encoding="utf-8")
    print(f"Fetched schema and meta from {repo}@{ref}")


def fetch_schema_pair(repo: str, ref: str) -> tuple[dict, dict]:
    errors = []
    for schema_path, meta_path in schema_source_paths(ref):
        schema_url = f"https://raw.githubusercontent.com/{repo}/{ref}/{schema_path}"
        meta_url = f"https://raw.githubusercontent.com/{repo}/{ref}/{meta_path}"
        try:
            return fetch_json(schema_url), fetch_json(meta_url)
        except RuntimeError as exc:
            errors.append(str(exc))

    attempted = "\n".join(f"- {error}" for error in errors)
    raise RuntimeError(f"Failed to fetch schema and meta from {repo}@{ref}. Attempts:\n{attempted}")


def schema_source_paths(ref: str) -> tuple[tuple[str, str], ...]:
    if re.fullmatch(r"refs/tags/schema-v\d+\.\d+\.\d+", ref):
        return (V1_SCHEMA_PATHS, LEGACY_SCHEMA_PATHS)
    return (LEGACY_SCHEMA_PATHS, V1_SCHEMA_PATHS)


def fetch_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 - trusted source configured by repo
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def _cached_ref() -> str | None:
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text(encoding="utf-8").strip() or None
    return None


if __name__ == "__main__":
    main()
