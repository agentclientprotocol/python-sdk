#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts import gen_meta, gen_schema, gen_signature  # noqa: E402  pylint: disable=wrong-import-position

SCHEMA_DIR = ROOT / "schema"

DEFAULT_REPO = "agentclientprotocol/agent-client-protocol"
LEGACY_SCHEMA_PATHS = ("schema/schema.unstable.json", "schema/meta.unstable.json")
V1_SCHEMA_PATHS = ("schema/v1/schema.unstable.json", "schema/v1/meta.unstable.json")
V2_SCHEMA_PATHS = ("schema/v2/schema.unstable.json", "schema/v2/meta.unstable.json")


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
        "--protocol-version",
        type=int,
        choices=(1, 2),
        default=int(os.environ.get("ACP_SCHEMA_PROTOCOL", "1")),
        help="Protocol bindings to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading schema files even when a version is provided.",
    )
    parser.set_defaults(format_output=True)
    parser.add_argument(
        "--no-format",
        dest="format_output",
        action="store_false",
        help="Skip formatting generated Python files after regeneration.",
    )
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
    protocol_version = args.protocol_version
    schema_dir = _schema_dir(protocol_version)
    schema_json = schema_dir / "schema.json"
    meta_json = schema_dir / "meta.json"
    should_download = _should_download(args, version, protocol_version)

    if should_download:
        ref = resolve_ref(version)
        download_schema(repo, ref, protocol_version)
    else:
        ref = resolve_ref(version) if version else _cached_ref(protocol_version)

    if not (schema_json.exists() and meta_json.exists()):
        print(
            f"{schema_dir.relative_to(ROOT)} is missing schema.json or meta.json; run with --version.", file=sys.stderr
        )
        sys.exit(1)

    gen_schema.generate_schema(protocol_version=protocol_version)
    gen_meta.generate_meta(protocol_version=protocol_version)
    if protocol_version == 1:
        gen_signature.gen_signature(ROOT / "src" / "acp")
    if args.format_output:
        format_generated_files(protocol_version)

    if ref:
        print(f"Generated schema using ref: {ref}")
    else:
        print("Generated schema using local schema files")


def format_generated_files(protocol_version: int) -> None:
    if protocol_version == 1:
        files = [
            ROOT / "src" / "acp" / "schema.py",
            ROOT / "src" / "acp" / "meta.py",
            ROOT / "src" / "acp" / "interfaces.py",
            ROOT / "src" / "acp" / "agent" / "connection.py",
            ROOT / "src" / "acp" / "client" / "connection.py",
        ]
    else:
        files = [
            ROOT / "src" / "acp" / "experimental" / "v2" / "schema.py",
            ROOT / "src" / "acp" / "experimental" / "v2" / "meta.py",
        ]
    subprocess.check_call([sys.executable, "-m", "ruff", "check", "--fix", *(str(path) for path in files)])  # noqa: S603
    subprocess.check_call([sys.executable, "-m", "ruff", "format", *(str(path) for path in files)])  # noqa: S603


def _should_download(args: argparse.Namespace, version: str | None, protocol_version: int) -> bool:
    schema_dir = _schema_dir(protocol_version)
    schema_json = schema_dir / "schema.json"
    meta_json = schema_dir / "meta.json"
    env_override = os.environ.get("ACP_SCHEMA_DOWNLOAD")
    if env_override is not None:
        return env_override.lower() in {"1", "true", "yes"}
    if args.no_download:
        return False
    if version:
        if not schema_json.exists() or not meta_json.exists():
            return True
        cached = _cached_ref(protocol_version)
        if args.force:
            return True
        return cached != resolve_ref(version)
    return not (schema_json.exists() and meta_json.exists())


def resolve_ref(version: str | None) -> str:
    if not version:
        return "refs/heads/main"
    if version.startswith("refs/"):
        return version
    if re.fullmatch(r"schema-v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version):
        return f"refs/tags/{version}"
    if re.fullmatch(r"v?\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version):
        value = version if version.startswith("v") else f"v{version}"
        return f"refs/tags/{value}"
    return f"refs/heads/{version}"


def download_schema(repo: str, ref: str, protocol_version: int = 1) -> None:
    schema_dir = _schema_dir(protocol_version)
    schema_json = schema_dir / "schema.json"
    meta_json = schema_dir / "meta.json"
    version_file = schema_dir / "VERSION"
    schema_dir.mkdir(parents=True, exist_ok=True)
    try:
        schema_data, meta_data = fetch_schema_pair(repo, ref, protocol_version)
    except RuntimeError as exc:  # pragma: no cover - network error path
        print(exc, file=sys.stderr)
        sys.exit(1)

    schema_json.write_text(json.dumps(schema_data, indent=2) + "\n", encoding="utf-8")
    meta_json.write_text(json.dumps(meta_data, indent=2) + "\n", encoding="utf-8")
    version_file.write_text(ref + "\n", encoding="utf-8")
    print(f"Fetched schema and meta from {repo}@{ref}")


def fetch_schema_pair(repo: str, ref: str, protocol_version: int = 1) -> tuple[dict, dict]:
    errors = []
    for schema_path, meta_path in schema_source_paths(ref, protocol_version):
        schema_url = f"https://raw.githubusercontent.com/{repo}/{ref}/{schema_path}"
        meta_url = f"https://raw.githubusercontent.com/{repo}/{ref}/{meta_path}"
        try:
            return fetch_json(schema_url), fetch_json(meta_url)
        except RuntimeError as exc:
            errors.append(str(exc))

    attempted = "\n".join(f"- {error}" for error in errors)
    raise RuntimeError(f"Failed to fetch schema and meta from {repo}@{ref}. Attempts:\n{attempted}")


def schema_source_paths(ref: str, protocol_version: int = 1) -> tuple[tuple[str, str], ...]:
    if protocol_version == 2:
        return (V2_SCHEMA_PATHS,)
    if re.fullmatch(r"refs/tags/schema-v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", ref):
        return (V1_SCHEMA_PATHS, LEGACY_SCHEMA_PATHS)
    return (LEGACY_SCHEMA_PATHS, V1_SCHEMA_PATHS)


def fetch_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 - trusted source configured by repo
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def _schema_dir(protocol_version: int) -> Path:
    if protocol_version == 1:
        return SCHEMA_DIR
    if protocol_version == 2:
        return SCHEMA_DIR / "v2"
    raise ValueError(f"Unsupported protocol version: {protocol_version}")


def _cached_ref(protocol_version: int) -> str | None:
    version_file = _schema_dir(protocol_version) / "VERSION"
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").strip() or None
    return None


if __name__ == "__main__":
    main()
