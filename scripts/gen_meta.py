#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ACP method metadata.")
    parser.add_argument("--protocol-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--check", action="store_true", help="Fail if the committed metadata is stale.")
    args = parser.parse_args()
    if not generate_meta(protocol_version=args.protocol_version, check=args.check):
        raise SystemExit(1)


def generate_meta(*, protocol_version: int = 1, check: bool = False) -> bool:
    _, out_py = _generation_paths(protocol_version)
    candidate = render_meta(protocol_version=protocol_version)
    current = out_py.read_text(encoding="utf-8") if out_py.exists() else ""
    if check:
        if current == candidate:
            return True
        print(
            "".join(
                difflib.unified_diff(
                    current.splitlines(keepends=True),
                    candidate.splitlines(keepends=True),
                    fromfile=str(out_py.relative_to(ROOT)),
                    tofile=f"{out_py.relative_to(ROOT)} (generated)",
                )
            ),
            end="",
        )
        return False
    out_py.parent.mkdir(parents=True, exist_ok=True)
    out_py.write_text(candidate, encoding="utf-8")
    return True


def render_meta(*, protocol_version: int = 1) -> str:
    schema_dir, out_py = _generation_paths(protocol_version)
    meta_json = schema_dir / "meta.json"
    version_file = schema_dir / "VERSION"
    if not meta_json.exists():
        raise SystemExit(f"{meta_json.relative_to(ROOT)} not found. Run gen_all.py first.")

    data = json.loads(meta_json.read_text("utf-8"))
    agent_methods = data.get("agentMethods", {})
    client_methods = data.get("clientMethods", {})
    version = data.get("version", 1)
    header_lines = [f"# Generated from {meta_json.relative_to(ROOT)}. Do not edit by hand."]
    if version_file.exists():
        ref = version_file.read_text("utf-8").strip()
        if ref:
            header_lines.append(f"# Schema ref: {ref}")

    source = (
        "\n".join(header_lines)
        + "\n"
        + f"AGENT_METHODS = {json.dumps(agent_methods, indent=4)}\n"
        + f"CLIENT_METHODS = {json.dumps(client_methods, indent=4)}\n"
        + f"PROTOCOL_VERSION = {int(version)}\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "ruff", "format", "--stdin-filename", str(out_py), "-"],
        input=source,
        text=True,
        capture_output=True,
        check=False,
        cwd=ROOT,
    )
    if result.returncode:
        raise RuntimeError(f"ruff format failed:\n{result.stderr}")
    return result.stdout


def _generation_paths(protocol_version: int) -> tuple[Path, Path]:
    if protocol_version == 1:
        return ROOT / "schema", ROOT / "src" / "acp" / "meta.py"
    if protocol_version == 2:
        return ROOT / "schema" / "v2", ROOT / "src" / "acp" / "experimental" / "v2" / "meta.py"
    raise ValueError(f"Unsupported protocol version: {protocol_version}")


if __name__ == "__main__":
    main()
