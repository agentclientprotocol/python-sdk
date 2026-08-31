#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from datamodel_code_generator import (
    DataModelType,
    Formatter,
    InputFileType,
    LiteralType,
    PythonVersion,
    generate,
)
from datamodel_code_generator.enums import NamingStrategy, VersionMode
from datamodel_code_generator.validators import ModelValidators, ValidatorDefinition, ValidatorMode
from pydantic.alias_generators import to_snake

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts._schema_semantics import (  # noqa: E402
    DEFAULT_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    SchemaSemantics,
    get_schema_semantics,
)

UNSIGNED_TYPE_MAPPINGS = (
    "integer+uint16=integer",
    "integer+uint32=integer",
    "integer+uint64=integer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ACP schema bindings.")
    parser.add_argument("--check", action="store_true", help="Fail if the committed bindings are stale.")
    parser.add_argument(
        "--protocol-version",
        type=int,
        choices=SUPPORTED_PROTOCOL_VERSIONS,
        default=DEFAULT_PROTOCOL_VERSION,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not generate_schema(check=args.check, protocol_version=args.protocol_version):
        raise SystemExit(1)


def generate_schema(*, check: bool = False, protocol_version: int = DEFAULT_PROTOCOL_VERSION) -> bool:
    semantics = get_schema_semantics(protocol_version)
    candidate = render_schema(semantics)
    schema_out = semantics.schema_out
    current = schema_out.read_text(encoding="utf-8") if schema_out.exists() else ""
    if check:
        if current == candidate:
            return True
        print(
            "".join(
                difflib.unified_diff(
                    current.splitlines(keepends=True),
                    candidate.splitlines(keepends=True),
                    fromfile=str(schema_out.relative_to(ROOT)),
                    tofile=f"{schema_out.relative_to(ROOT)} (generated)",
                )
            ),
            end="",
        )
        return False
    schema_out.parent.mkdir(parents=True, exist_ok=True)
    schema_out.write_text(candidate, encoding="utf-8")
    return True


def render_schema(semantics: SchemaSemantics) -> str:
    """Generate bindings directly from the pinned JSON Schema."""
    schema_json = semantics.schema_json
    if not schema_json.exists():
        raise FileNotFoundError(f"{schema_json.relative_to(ROOT)} is missing; fetch a pinned schema release first")

    schema = json.loads(schema_json.read_text(encoding="utf-8"))
    generated = generate(
        schema,
        input_file_type=InputFileType.JsonSchema,
        custom_file_header=_build_header(schema_json, semantics.version_file),
        target_python_version=PythonVersion.PY_310,
        collapse_root_models=True,
        skip_root_model=True,
        output_model_type=DataModelType.PydanticV2BaseModel,
        base_class="acp._schema_base.BaseModel",
        use_specialized_enum=False,
        use_standard_collections=False,
        use_union_operator=False,
        additional_imports=["enum.Enum"],
        enum_field_as_literal=LiteralType.All,
        use_one_literal_as_default=True,
        validators=_build_validators_config(schema),
        formatters=[Formatter.BUILTIN],
        infer_union_variant_names=True,
        naming_strategy=NamingStrategy.PrimaryFirst,
        model_name_map=semantics.model_name_map,
        strict_refs=True,
        schema_version="2020-12",
        schema_version_mode=VersionMode.Strict,
        type_mappings=list(UNSIGNED_TYPE_MAPPINGS),
        generate_schema_validators=True,
        use_annotated=True,
        field_constraints=True,
        use_field_description=True,
        snake_case_field=True,
    )
    if not isinstance(generated, str):
        raise TypeError("Schema generation did not produce a single Python module")
    source = generated.rstrip()
    if semantics.compatibility_aliases:
        source = f"{source}\n\n\n{semantics.compatibility_aliases}"
    return _format_python(f"{source}\n", semantics.schema_out)


def _build_validators_config(schema: dict[str, Any]) -> dict[str, ModelValidators]:
    validators: dict[str, list[ValidatorDefinition]] = {
        "InitializeRequest": [
            ValidatorDefinition(
                field="protocol_version",
                function="acp._deserialize.coerce_protocol_version",
                mode=ValidatorMode.BEFORE,
            )
        ]
    }
    for class_name, definition in schema.get("$defs", {}).items():
        if not isinstance(definition, dict):
            continue
        default_fields, skip_fields = _deserialize_field_specs(definition)
        definitions = validators.setdefault(class_name, [])
        definitions.extend(
            ValidatorDefinition(fields=sorted(fields), function=function, mode=ValidatorMode.WRAP)
            for fields, function in (
                (default_fields, "acp._deserialize.use_default_on_error"),
                (skip_fields, "acp._deserialize.skip_invalid_items"),
            )
            if fields
        )
    return {
        class_name: ModelValidators(validators=definitions)
        for class_name, definitions in validators.items()
        if definitions
    }


def _deserialize_field_specs(definition: dict[str, Any]) -> tuple[list[str], list[str]]:
    required = set(definition.get("required", []))
    use_default: list[str] = []
    skip: list[str] = []
    for property_name, property_schema in definition.get("properties", {}).items():
        if not isinstance(property_schema, dict) or property_name == "_meta":
            continue
        field_name = to_snake(property_name)
        if property_schema.get("x-deserialize-skip-invalid-items"):
            skip.append(field_name)
        elif property_schema.get("x-deserialize-default-on-error"):
            if property_name in required:
                raise ValueError(f"{property_name!r} requests default-on-error but is required")
            use_default.append(field_name)
    return use_default, skip


def _build_header(schema_json: Path, version_file: Path) -> str:
    lines = [f"# Generated from {schema_json.relative_to(ROOT)}. Do not edit by hand."]
    if version_file.exists() and (ref := version_file.read_text(encoding="utf-8").strip()):
        lines.append(f"# Schema ref: {ref}")
    return "\n".join(lines)


def _format_python(source: str, schema_out: Path) -> str:
    commands = (
        ("check", "--fix"),
        ("format",),
    )
    for arguments in commands:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "ruff", *arguments, "--stdin-filename", str(schema_out), "-"],
            input=source,
            text=True,
            capture_output=True,
            check=False,
            cwd=ROOT,
        )
        if result.returncode:
            raise RuntimeError(f"ruff {' '.join(arguments)} failed:\n{result.stderr}")
        source = result.stdout
    return source


if __name__ == "__main__":
    main()
