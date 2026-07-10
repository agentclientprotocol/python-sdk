#!/usr/bin/env python3
from __future__ import annotations

import ast
import copy
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schema"
SCHEMA_JSON = SCHEMA_DIR / "schema.json"
VERSION_FILE = SCHEMA_DIR / "VERSION"
SCHEMA_OUT = ROOT / "src" / "acp" / "schema.py"

# Pattern caches used when post-processing generated schema.
FIELD_DECLARATION_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*:")
DESCRIPTION_PATTERN = re.compile(
    r"description\s*=\s*(?P<prefix>[rRbBuU]*)?(?P<quote>'''|\"\"\"|'|\")(?P<value>.*?)(?P=quote)",
    re.DOTALL,
)

STDIO_TYPE_LITERAL = 'Literal["2#-datamodel-code-generator-#-object-#-special-#"]'
MODELS_TO_REMOVE = [
    "AgentClientProtocol",
    "AgentClientProtocol1",
    "AgentClientProtocol2",
    "AgentClientProtocol3",
    "AgentClientProtocol4",
    "AgentClientProtocol5",
    "AgentClientProtocol6",
    "AgentClientProtocol7",
]

# Map of numbered classes produced by datamodel-code-generator to descriptive names.
# Keep this in sync with the Rust/TypeScript SDK nomenclature.
RENAME_MAP: dict[str, str] = {
    "AgentResponse1": "AgentResponseMessage",
    "AgentResponse2": "AgentErrorMessage",
    "ClientResponse1": "ClientResponseMessage",
    "ClientResponse2": "ClientErrorMessage",
    "ContentBlock1": "TextContentBlock",
    "ContentBlock2": "ImageContentBlock",
    "ContentBlock3": "AudioContentBlock",
    "ContentBlock4": "ResourceContentBlock",
    "ContentBlock5": "EmbeddedResourceContentBlock",
    "McpServer1": "HttpMcpServer",
    "McpServer2": "SseMcpServer",
    "McpServer3": "AcpMcpServer",
    "RequestPermissionOutcome1": "DeniedOutcome",
    "RequestPermissionOutcome2": "AllowedOutcome",
    "AuthMethod1": "EnvVarAuthMethod",
    "AuthMethod2": "TerminalAuthMethod",
    "SessionConfigOption1": "SessionConfigOptionSelect",
    "SessionConfigOption2": "SessionConfigOptionBoolean",
    "SetSessionConfigOptionRequest1": "SetSessionConfigOptionBooleanRequest",
    "SetSessionConfigOptionRequest2": "SetSessionConfigOptionSelectRequest",
    "SessionUpdate1": "UserMessageChunk",
    "SessionUpdate2": "AgentMessageChunk",
    "SessionUpdate3": "AgentThoughtChunk",
    "SessionUpdate4": "ToolCallStart",
    "SessionUpdate5": "ToolCallProgress",
    "SessionUpdate6": "AgentPlanUpdate",
    "SessionUpdate7": "AgentPlanContentUpdate",
    "SessionUpdate8": "AgentPlanRemovedUpdate",
    "SessionUpdate9": "AvailableCommandsUpdate",
    "SessionUpdate10": "CurrentModeUpdate",
    "SessionUpdate11": "ConfigOptionUpdate",
    "SessionUpdate12": "SessionInfoUpdate",
    "SessionUpdate13": "UsageUpdate",
    "PlanUpdateContent1": "PlanUpdateItems",
    "PlanUpdateContent2": "PlanUpdateFile",
    "PlanUpdateContent3": "PlanUpdateMarkdown",
    "ToolCallContent1": "ContentToolCallContent",
    "ToolCallContent2": "FileEditToolCallContent",
    "ToolCallContent3": "TerminalToolCallContent",
    "CreateElicitationRequest1": "CreateFormSessionElicitationRequest",
    "CreateElicitationRequest2": "CreateFormRequestElicitationRequest",
    "CreateElicitationRequest3": "CreateUrlSessionElicitationRequest",
    "CreateElicitationRequest4": "CreateUrlRequestElicitationRequest",
    "CreateElicitationRequest5": "CreateOtherElicitationRequest",
    "CreateElicitationResponse1": "AcceptElicitationResponse",
    "CreateElicitationResponse2": "DeclineElicitationResponse",
    "CreateElicitationResponse3": "CancelElicitationResponse",
    "CreateElicitationResponse4": "OtherElicitationResponse",
    "ElicitationFormMode1": "ElicitationFormSessionMode",
    "ElicitationFormMode2": "ElicitationFormRequestMode",
    "ElicitationPropertySchema1": "ElicitationStringPropertySchema",
    "ElicitationPropertySchema2": "ElicitationNumberPropertySchema",
    "ElicitationPropertySchema3": "ElicitationIntegerPropertySchema",
    "ElicitationPropertySchema4": "ElicitationBooleanPropertySchema",
    "ElicitationPropertySchema5": "ElicitationMultiSelectPropertySchema",
    "ElicitationPropertySchema6": "ElicitationOtherPropertySchema",
    "MultiSelectItems1": "StringMultiSelectItems",
    "MultiSelectItems2": "OtherMultiSelectItems",
    "ElicitationUrlMode1": "ElicitationUrlSessionMode",
    "ElicitationUrlMode2": "ElicitationUrlRequestMode",
    "NesSuggestion1": "NesEditSuggestionVariant",
    "NesSuggestion2": "NesJumpSuggestionVariant",
    "NesSuggestion3": "NesRenameSuggestionVariant",
    "NesSuggestion4": "NesSearchAndReplaceSuggestionVariant",
}

# Extensible ("custom or future") unions: known const-tagged variants plus a
# catch-all member tagged `"title": "other"`. _normalize_catchall_unions strips the
# discriminator and the catch-all's `not` clause so datamodel-codegen produces a plain
# union; the exclusion is restored at runtime by a field_validator injected into the
# catch-all class, so a malformed known variant fails instead of silently parsing as
# custom (mirrors the TypeScript SDK's excludeKnownTags). Maps union def name ->
# catch-all class name; the set is asserted against the schema in
# _validate_schema_alignment.
EXTENSIBLE_UNIONS: dict[str, str] = {
    "CreateElicitationRequest": "CreateOtherElicitationRequest",
    "CreateElicitationResponse": "OtherElicitationResponse",
    "ElicitationPropertySchema": "ElicitationOtherPropertySchema",
    "MultiSelectItems": "OtherMultiSelectItems",
}

ENUM_LITERAL_MAP: dict[str, tuple[str, ...]] = {
    "PermissionOptionKind": (
        "allow_once",
        "allow_always",
        "reject_once",
        "reject_always",
    ),
    "PlanEntryPriority": ("high", "medium", "low"),
    "PlanEntryStatus": ("pending", "in_progress", "completed"),
    "StopReason": ("end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"),
    "ToolCallStatus": ("pending", "in_progress", "completed", "failed"),
    "ToolKind": ("read", "edit", "delete", "move", "search", "execute", "think", "fetch", "switch_mode", "other"),
}

FIELD_TYPE_OVERRIDES: tuple[tuple[str, str, str, bool], ...] = (
    ("PermissionOption", "kind", "PermissionOptionKind", False),
    ("PlanEntry", "priority", "PlanEntryPriority", False),
    ("PlanEntry", "status", "PlanEntryStatus", False),
    ("PromptResponse", "stop_reason", "StopReason", False),
    ("ToolCall", "kind", "ToolKind", True),
    ("ToolCall", "status", "ToolCallStatus", True),
    ("ToolCallUpdate", "kind", "ToolKind", True),
    ("ToolCallUpdate", "status", "ToolCallStatus", True),
)


@dataclass(frozen=True)
class FieldValidatorInjection:
    """A generated field validator that should be appended to one schema class."""

    class_name: str
    field_name: str
    method_name: str
    argument_name: str
    return_type: str
    comment_lines: tuple[str, ...]
    body_lines: tuple[str, ...]

    def render(self) -> str:
        lines = [
            f'@field_validator("{self.field_name}", mode="before")',
            "@classmethod",
            f"def {self.method_name}(cls, {self.argument_name}: Any) -> {self.return_type}:",
        ]
        lines.extend(f"    # {line}" for line in self.comment_lines)
        lines.extend(f"    {line}" for line in self.body_lines)
        return "\n".join(lines)


DEFAULT_VALUE_OVERRIDES: tuple[tuple[str, str, str], ...] = (
    ("AgentCapabilities", "mcp_capabilities", "McpCapabilities()"),
    ("AgentCapabilities", "session_capabilities", "SessionCapabilities()"),
    (
        "AgentCapabilities",
        "prompt_capabilities",
        "PromptCapabilities()",
    ),
    ("ClientCapabilities", "fs", "FileSystemCapabilities()"),
    ("ClientCapabilities", "terminal", "False"),
    (
        "InitializeRequest",
        "client_capabilities",
        "ClientCapabilities()",
    ),
    (
        "InitializeResponse",
        "agent_capabilities",
        "AgentCapabilities()",
    ),
)

# Classes that need a field_validator injected after generation.
CLASS_VALIDATOR_INJECTIONS: tuple[FieldValidatorInjection, ...] = (
    FieldValidatorInjection(
        class_name="InitializeRequest",
        field_name="protocol_version",
        method_name="_coerce_protocol_version",
        argument_name="value",
        return_type="int",
        comment_lines=(
            'Some clients (e.g. Zed) send a date string like "2024-11-05" instead',
            "of an integer. The Rust SDK treats legacy strings as version 0; this",
            "SDK maps unparsable values to 1 so the connection is not rejected.",
            "See: https://github.com/agentclientprotocol/rust-sdk/blob/main/crates/agent-client-protocol-schema/src/version.rs",
        ),
        body_lines=(
            "if isinstance(value, int):",
            "    return value",
            "try:",
            "    return int(value)",
            "except (TypeError, ValueError):",
            "    return 1",
        ),
    ),
)


@dataclass(frozen=True)
class _ProcessingStep:
    """A named transformation applied to the generated schema content."""

    name: str
    apply: Callable[[str], str]


def main() -> None:
    generate_schema()


def generate_schema() -> None:
    if not SCHEMA_JSON.exists():
        print(
            "Schema file missing. Ensure schema/schema.json exists (run gen_all.py --version to download).",
            file=sys.stderr,
        )
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        codegen_input = Path(tmp_dir) / "schema.codegen.json"
        codegen_input.write_text(json.dumps(_preprocess_schema_for_codegen(_load_schema()), indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "datamodel_code_generator",
            "--input",
            str(codegen_input),
            "--input-file-type",
            "jsonschema",
            "--output",
            str(SCHEMA_OUT),
            "--target-python-version",
            "3.12",
            "--collapse-root-models",
            "--output-model-type",
            "pydantic_v2.BaseModel",
            "--use-annotated",
            "--snake-case-field",
        ]

        subprocess.check_call(cmd)  # noqa: S603
    warnings = postprocess_generated_schema(SCHEMA_OUT)
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)


def _load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))


COMBINATOR_KEYS = ("oneOf", "anyOf")


def _preprocess_schema_for_codegen(schema: dict[str, Any]) -> dict[str, Any]:
    schema = _normalize_catchall_unions(schema)
    defs = schema.get("$defs", {})
    return _distribute_composed_object_schemas(schema, defs)


def _normalize_catchall_unions(node: Any) -> Any:
    # ACP "custom or future" unions include a member tagged `"title": "other"` whose
    # discriminator (type/mode/action) is a free-form string. datamodel-codegen cannot
    # put that in a discriminated union, so it emits `#-special-#` placeholder literals.
    # Drop the discriminator (the union is then validated structurally) and collapse the
    # catch-all to a permissive object so unknown variants round-trip their raw payload.
    if isinstance(node, list):
        return [_normalize_catchall_unions(item) for item in node]
    if not isinstance(node, dict):
        return node

    transformed = {key: _normalize_catchall_unions(value) for key, value in node.items()}
    for combinator in COMBINATOR_KEYS:
        members = transformed.get(combinator)
        if not isinstance(members, list):
            continue
        if not any(isinstance(member, dict) and member.get("title") == "other" for member in members):
            continue
        transformed.pop("discriminator", None)
        transformed[combinator] = [
            _collapse_catchall_member(member) if isinstance(member, dict) and member.get("title") == "other" else member
            for member in members
        ]
    return transformed


def _collapse_catchall_member(member: dict[str, Any]) -> dict[str, Any]:
    collapsed: dict[str, Any] = {"type": "object", "additionalProperties": True}
    for key in ("title", "description", "properties", "required"):
        if key in member:
            collapsed[key] = member[key]
    return collapsed


def _distribute_composed_object_schemas(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_distribute_composed_object_schemas(item, defs) for item in node]
    if not isinstance(node, dict):
        return node

    transformed = {key: _distribute_composed_object_schemas(value, defs) for key, value in node.items()}
    for combinator in COMBINATOR_KEYS:
        if combinator not in transformed or "properties" not in transformed:
            continue
        result = {combinator: _expand_composed_object_variants(transformed, defs)}
        for key in ("title", "description", "discriminator"):
            if key in transformed:
                result[key] = transformed[key]
        return result
    return transformed


def _expand_composed_object_variants(node: dict[str, Any], defs: dict[str, Any]) -> list[Any]:
    for combinator in COMBINATOR_KEYS:
        if combinator not in node or "properties" not in node:
            continue

        common_schema = _without_combinators(node)
        expanded: list[Any] = []
        for option in node[combinator]:
            for variant in _expand_allof_union_refs(option, defs):
                expanded.append(_merge_object_schema(common_schema, variant) if isinstance(variant, dict) else variant)
        return expanded

    return _expand_allof_union_refs(node, defs)


def _expand_allof_union_refs(node: Any, defs: dict[str, Any]) -> list[Any]:
    if not isinstance(node, dict):
        return [node]

    variants = [{key: copy.deepcopy(value) for key, value in node.items() if key != "allOf"}]
    for item in node.get("allOf", []):
        ref_name = _local_def_ref_name(item.get("$ref")) if isinstance(item, dict) else None
        ref_schema = defs.get(ref_name) if ref_name else None
        if isinstance(ref_schema, dict) and any(key in ref_schema for key in COMBINATOR_KEYS):
            ref_variants = _expand_composed_object_variants(ref_schema, defs)
        else:
            ref_variants = [item]

        variants = [
            _merge_object_schema(variant, ref_variant) if isinstance(ref_variant, dict) else variant
            for variant in variants
            for ref_variant in ref_variants
        ]
    return variants


def _without_combinators(node: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in node.items()
        if key not in COMBINATOR_KEYS and key != "discriminator"
    }


def _local_def_ref_name(ref: Any) -> str | None:
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return ref.rsplit("/", 1)[-1]
    return None


def _pop_ref_as_allof(schema: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema = copy.deepcopy(schema)
    if "$ref" not in schema:
        return schema, []
    return schema, [{"$ref": schema.pop("$ref")}]


def _merge_object_schema(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left, left_refs = _pop_ref_as_allof(left)
    right, right_refs = _pop_ref_as_allof(right)
    merged: dict[str, Any] = {}

    for key in set(left) | set(right):
        if key in COMBINATOR_KEYS or key in {"allOf", "discriminator"}:
            continue
        if key == "properties":
            merged[key] = {**left.get(key, {}), **right.get(key, {})}
        elif key == "required":
            required = []
            for item in left.get(key, []) + right.get(key, []):
                if item not in required:
                    required.append(item)
            if required:
                merged[key] = required
        elif key in right:
            merged[key] = right[key]
        else:
            merged[key] = left[key]

    all_of = left_refs + left.get("allOf", []) + right_refs + right.get("allOf", [])
    if all_of:
        merged["allOf"] = all_of
    return merged


def _required_nullable_fields(schema: dict[str, Any]) -> dict[str, list[str]]:
    defs = schema.get("$defs", {})
    fields: dict[str, list[str]] = {}
    for class_name, definition in defs.items():
        if not isinstance(definition, dict):
            continue

        required = set(definition.get("required", []))
        if not required:
            continue

        properties = definition.get("properties", {})
        nullable_fields = [
            _schema_field_name(property_name)
            for property_name in sorted(required)
            if _schema_allows_null(properties.get(property_name), defs)
        ]
        if nullable_fields:
            fields[class_name] = nullable_fields
    return fields


def _schema_allows_null(node: Any, defs: dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False

    schema_type = node.get("type")
    if schema_type == "null" or (isinstance(schema_type, list) and "null" in schema_type):
        return True

    for combinator in COMBINATOR_KEYS:
        if any(_schema_allows_null(option, defs) for option in node.get(combinator, [])):
            return True

    ref_name = _local_def_ref_name(node.get("$ref"))
    if ref_name is not None:
        return _schema_allows_null(defs.get(ref_name), defs)

    return any(_schema_allows_null(option, defs) for option in node.get("allOf", []))


def _schema_field_name(name: str) -> str:
    if name.startswith("_"):
        return "field" + name
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def postprocess_generated_schema(output_path: Path) -> list[str]:
    if not output_path.exists():
        raise RuntimeError(f"Generated schema not found at {output_path}")

    raw_content = output_path.read_text(encoding="utf-8")
    header_block = _build_header_block()

    content = _strip_existing_header(raw_content)
    content = _remove_unused_models(content)
    content, leftover_classes = _rename_numbered_models(content)

    processing_steps: tuple[_ProcessingStep, ...] = (
        _ProcessingStep("apply field overrides", _apply_field_overrides),
        _ProcessingStep("apply default overrides", _apply_default_overrides),
        _ProcessingStep("restore required nullable fields", _restore_required_nullable_fields),
        _ProcessingStep("attach description comments", _add_description_comments),
        _ProcessingStep("ensure custom BaseModel", _ensure_custom_base_model),
        _ProcessingStep("inject field validators", _inject_field_validators),
        _ProcessingStep("inject schema aliases", _inject_schema_aliases),
    )

    for step in processing_steps:
        content = step.apply(content)

    missing_targets = _find_missing_targets(content)

    content = _inject_enum_aliases(content)
    final_content = header_block + content.rstrip() + "\n"
    if not final_content.endswith("\n"):
        final_content += "\n"
    output_path.write_text(final_content, encoding="utf-8")

    warnings: list[str] = []
    if leftover_classes:
        warnings.append(
            "Unrenamed schema models detected: "
            + ", ".join(leftover_classes)
            + ". Update RENAME_MAP in scripts/gen_schema.py."
        )
    if missing_targets:
        warnings.append(
            "Renamed schema targets not found after generation: "
            + ", ".join(sorted(missing_targets))
            + ". Check RENAME_MAP or upstream schema changes."
        )
    warnings.extend(_validate_schema_alignment())

    return warnings


def _build_header_block() -> str:
    header_lines = ["# Generated from schema/schema.json. Do not edit by hand."]
    if VERSION_FILE.exists():
        ref = VERSION_FILE.read_text(encoding="utf-8").strip()
        if ref:
            header_lines.append(f"# Schema ref: {ref}")
    return "\n".join(header_lines) + "\n\n"


def _strip_existing_header(content: str) -> str:
    existing_header = re.match(r"(#.*\n)+", content)
    if existing_header:
        return content[existing_header.end() :].lstrip("\n")
    return content.lstrip("\n")


def _rename_numbered_models(content: str) -> tuple[str, list[str]]:
    renamed = content
    for old, new in sorted(RENAME_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(new)}\b", renamed) is not None:
            renamed = re.sub(rf"\b{re.escape(new)}\b", f"_{new}", renamed)
        pattern = re.compile(rf"\b{re.escape(old)}\b")
        renamed = pattern.sub(new, renamed)

    leftover_class_pattern = re.compile(r"^class (\w+\d+)\(", re.MULTILINE)
    leftover_classes = sorted(set(leftover_class_pattern.findall(renamed)))
    return renamed, leftover_classes


def _find_missing_targets(content: str) -> list[str]:
    missing: list[str] = []
    for new_name in RENAME_MAP.values():
        pattern = re.compile(rf"^class {re.escape(new_name)}\(", re.MULTILINE)
        if not pattern.search(content):
            missing.append(new_name)
    return missing


def _validate_schema_alignment() -> list[str]:
    warnings: list[str] = []
    if not SCHEMA_JSON.exists():
        warnings.append("schema/schema.json missing; unable to validate enum aliases.")
        return warnings

    try:
        schema_enums = _load_schema_enum_literals()
    except json.JSONDecodeError as exc:
        warnings.append(f"Failed to parse schema/schema.json: {exc}")
        return warnings

    for enum_name, expected_values in ENUM_LITERAL_MAP.items():
        schema_values = schema_enums.get(enum_name)
        if schema_values is None:
            warnings.append(
                f"Enum '{enum_name}' not found in schema.json; update ENUM_LITERAL_MAP or investigate schema changes."
            )
            continue
        if tuple(schema_values) != expected_values:
            warnings.append(
                f"Enum mismatch for '{enum_name}': schema.json -> {schema_values}, generated aliases -> {expected_values}"
            )

    detected_unions = _detect_extensible_unions()
    if detected_unions != set(EXTENSIBLE_UNIONS):
        warnings.append(
            f"Extensible union drift: schema defines {sorted(detected_unions)}, "
            f"EXTENSIBLE_UNIONS lists {sorted(EXTENSIBLE_UNIONS)}. Update EXTENSIBLE_UNIONS, the "
            "RENAME_MAP catch-all names, and the alias template together."
        )
    return warnings


def _detect_extensible_unions() -> set[str]:
    defs = _load_schema().get("$defs", {})
    detected: set[str] = set()
    for name, definition in defs.items():
        if not isinstance(definition, dict) or "discriminator" not in definition:
            continue
        members = definition.get("anyOf") or definition.get("oneOf") or []
        if any(isinstance(member, dict) and member.get("title") == "other" for member in members):
            detected.add(name)
    return detected


def _load_schema_enum_literals() -> dict[str, tuple[str, ...]]:
    schema_data = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    defs = schema_data.get("$defs", {})
    enum_literals: dict[str, tuple[str, ...]] = {}

    for name, definition in defs.items():
        values: list[str] = []
        if "enum" in definition:
            values = [str(item) for item in definition["enum"]]
        elif "oneOf" in definition:
            values = [
                str(option["const"])
                for option in definition.get("oneOf", [])
                if isinstance(option, dict) and "const" in option
            ]
        if values:
            enum_literals[name] = tuple(values)

    return enum_literals


def _ensure_custom_base_model(content: str) -> str:
    if "class BaseModel(_BaseModel):" in content:
        return content
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if not line.startswith("from pydantic import "):
            continue
        imports = [part.strip() for part in line[len("from pydantic import ") :].split(",")]
        has_alias = any(part == "BaseModel as _BaseModel" for part in imports)
        has_config = any(part == "ConfigDict" for part in imports)
        new_imports = []
        for part in imports:
            if part == "BaseModel":
                new_imports.append("BaseModel as _BaseModel")
                has_alias = True
            else:
                new_imports.append(part)
        if not has_alias:
            new_imports.append("BaseModel as _BaseModel")
        if not has_config:
            new_imports.append("ConfigDict")
        lines[idx] = "from pydantic import " + ", ".join(new_imports)
        to_insert = textwrap.dedent("""\
            class BaseModel(_BaseModel):
                model_config = ConfigDict(populate_by_name=True)

                def __getattr__(self, item: str) -> Any:
                    if item.lower() != item:
                        snake_cased = "".join("_" + c.lower() if c.isupper() and i > 0 else c.lower() for i, c in enumerate(item))
                        return getattr(self, snake_cased)
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")
        """)
        insert_idx = idx + 1
        lines.insert(insert_idx, "")
        for offset, line in enumerate(to_insert.splitlines(), 1):
            lines.insert(insert_idx + offset, line)
        break
    return "\n".join(lines) + "\n"


def _ensure_pydantic_import(content: str, name: str) -> str:
    """Add *name* to the ``from pydantic import ...`` line if not already present."""
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if not line.startswith("from pydantic import "):
            continue
        imports = [part.strip() for part in line[len("from pydantic import ") :].split(",")]
        if name not in imports:
            imports.append(name)
            lines[idx] = "from pydantic import " + ", ".join(imports)
        return "\n".join(lines) + "\n"
    return content


def _extensible_union_excluded_tags(union_def: dict[str, Any], discriminator: str) -> tuple[str, ...]:
    members = union_def.get("anyOf") or union_def.get("oneOf") or []
    other = next((member for member in members if isinstance(member, dict) and member.get("title") == "other"), None)
    if other is None:
        return ()
    tags: list[str] = []
    for excluded in other.get("not", {}).get("anyOf", []):
        const = excluded.get("properties", {}).get(discriminator, {}).get("const")
        if isinstance(const, str) and const not in tags:
            tags.append(const)
    return tuple(tags)


def _catchall_exclusion_injections() -> list[FieldValidatorInjection]:
    defs = _load_schema().get("$defs", {})
    injections: list[FieldValidatorInjection] = []
    for union_name, catchall_class in EXTENSIBLE_UNIONS.items():
        union_def = defs.get(union_name)
        if not isinstance(union_def, dict):
            continue
        discriminator = union_def.get("discriminator", {}).get("propertyName")
        if not discriminator:
            continue
        tags = _extensible_union_excluded_tags(union_def, discriminator)
        if not tags:
            continue
        field = _schema_field_name(discriminator)
        injections.append(
            FieldValidatorInjection(
                class_name=catchall_class,
                field_name=field,
                method_name=f"_reject_known_{field}",
                argument_name="value",
                return_type="Any",
                comment_lines=(
                    "Restore the schema's `not` clause dropped for codegen: reject the known",
                    "variants' discriminator values so a malformed known variant fails instead",
                    "of silently parsing as this catch-all.",
                ),
                body_lines=(
                    f"if value in {tags!r}:",
                    f'    raise ValueError("{field} value is reserved by a known variant")',
                    "return value",
                ),
            )
        )
    return injections


def _inject_field_validators(content: str) -> str:
    """Inject field_validator methods for CLASS_VALIDATOR_INJECTIONS and catch-all exclusions."""
    for injection in (*CLASS_VALIDATOR_INJECTIONS, *_catchall_exclusion_injections()):
        content = _ensure_pydantic_import(content, "field_validator")

        class_pattern = re.compile(
            rf"(class {injection.class_name}\(BaseModel\):)(.*?)(?=\nclass |\Z)",
            re.DOTALL,
        )

        def _append_validator(
            match: re.Match[str],
            _injection: FieldValidatorInjection = injection,
        ) -> str:
            header, block = match.group(1), match.group(2)
            indented = "\n" + textwrap.indent(_injection.render(), "    ")
            return header + block + indented + "\n"

        content, count = class_pattern.subn(_append_validator, content, count=1)
        if count == 0:
            print(
                f"Warning: class {injection.class_name} not found for validator injection",
                file=sys.stderr,
            )
    return content


def _inject_schema_aliases(content: str) -> str:
    if "CreateElicitationRequest = Union[" in content:
        return content

    aliases = textwrap.dedent("""\
        ElicitationMode = Union[
            ElicitationFormSessionMode,
            ElicitationFormRequestMode,
            ElicitationUrlSessionMode,
            ElicitationUrlRequestMode,
        ]
        CreateFormElicitationRequest = Union[
            CreateFormSessionElicitationRequest,
            CreateFormRequestElicitationRequest,
        ]
        CreateUrlElicitationRequest = Union[
            CreateUrlSessionElicitationRequest,
            CreateUrlRequestElicitationRequest,
        ]
        CreateElicitationRequest = Union[
            CreateFormElicitationRequest,
            CreateUrlElicitationRequest,
            CreateOtherElicitationRequest,
        ]
        CreateElicitationResponse = Union[
            AcceptElicitationResponse,
            DeclineElicitationResponse,
            CancelElicitationResponse,
            OtherElicitationResponse,
        ]
    """)
    pattern = re.compile(
        r"^(class CreateFormRequestElicitationRequest\([\s\S]*?\):[\s\S]*?)(?=^class \w+\(|\Z)",
        re.MULTILINE,
    )
    content, count = pattern.subn(lambda match: match.group(1).rstrip() + "\n\n" + aliases + "\n", content, count=1)
    if count == 0:
        print("Warning: failed to insert schema aliases", file=sys.stderr)
    return content


def _restore_required_nullable_fields(content: str, schema: dict[str, Any] | None = None) -> str:
    schema = _load_schema() if schema is None else schema
    for class_name, field_names in _required_nullable_fields(schema).items():
        class_pattern = re.compile(
            rf"(class {re.escape(class_name)}\([^)]*\):)(.*?)(?=\nclass |\Z)",
            re.DOTALL,
        )

        def restore_block(match: re.Match[str], _field_names: list[str] = field_names) -> str:
            header, block = match.group(1), match.group(2)
            for field_name in _field_names:
                field_pattern = re.compile(rf"(\n\s+{re.escape(field_name)}:\s+Annotated\[[\s\S]*?\n\s+\]\s*)=\s*None")
                block = field_pattern.sub(r"\1", block, count=1)
            return header + block

        content = class_pattern.sub(restore_block, content, count=1)
    return content


def _apply_field_overrides(content: str) -> str:
    for class_name, field_name, new_type, optional in FIELD_TYPE_OVERRIDES:
        if optional:
            pattern = re.compile(
                rf"(class {class_name}\(BaseModel\):.*?\n\s+{field_name}:\s+Annotated\[\s*)Optional\[str],",
                re.DOTALL,
            )
            content, count = pattern.subn(rf"\1Optional[{new_type}],", content)
        else:
            pattern = re.compile(
                rf"(class {class_name}\(BaseModel\):.*?\n\s+{field_name}:\s+Annotated\[\s*)str,",
                re.DOTALL,
            )
            content, count = pattern.subn(rf"\1{new_type},", content)
        if count == 0:
            print(
                f"Warning: failed to apply type override for {class_name}.{field_name} -> {new_type}",
                file=sys.stderr,
            )
    return content


def _apply_default_overrides(content: str) -> str:
    for class_name, field_name, replacement in DEFAULT_VALUE_OVERRIDES:
        class_pattern = re.compile(
            rf"(class {class_name}\(BaseModel\):)(.*?)(?=\nclass |\Z)",
            re.DOTALL,
        )

        def replace_block(
            match: re.Match[str],
            _field_name: str = field_name,
            _replacement: str = replacement,
            _class_name: str = class_name,
        ) -> str:
            header, block = match.group(1), match.group(2)
            field_patterns: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], str]], ...] = (
                (
                    re.compile(
                        rf"(\n\s+{_field_name}:.*?\]\s*=\s*)([\s\S]*?)(?=\n\s{{4}}[A-Za-z_]|$)",
                        re.DOTALL,
                    ),
                    lambda m, _rep=_replacement: m.group(1) + _rep,
                ),
                (
                    re.compile(
                        rf"(\n\s+{_field_name}:[^\n]*=)\s*([^\n]+)",
                        re.MULTILINE,
                    ),
                    lambda m, _rep=_replacement: m.group(1) + " " + _rep,
                ),
            )
            for pattern, replacer in field_patterns:
                new_block, count = pattern.subn(replacer, block, count=1)
                if count:
                    return header + new_block
            print(
                f"Warning: failed to override default for {_class_name}.{_field_name}",
                file=sys.stderr,
            )
            return match.group(0)

        content, count = class_pattern.subn(replace_block, content, count=1)
        if count == 0:
            print(
                f"Warning: class {class_name} not found for default override on {field_name}",
                file=sys.stderr,
            )
    return content


def _add_description_comments(content: str) -> str:
    lines = content.splitlines()
    new_lines: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if indent == 4 and FIELD_DECLARATION_PATTERN.match(stripped or ""):
            block_lines, next_index = _collect_field_block(lines, index, indent)
            block_text = "\n".join(block_lines)
            description = _extract_description(block_text)

            if description:
                indent_str = " " * indent
                comment_lines = [
                    f"{indent_str}# {comment_line}" if comment_line else f"{indent_str}#"
                    for comment_line in description.splitlines()
                ]
                if comment_lines:
                    new_lines.extend(comment_lines)

            new_lines.extend(block_lines)
            index = next_index
            continue

        new_lines.append(line)
        index += 1

    return "\n".join(new_lines)


def _collect_field_block(lines: list[str], start: int, indent: int) -> tuple[list[str], int]:
    block: list[str] = []
    index = start

    while index < len(lines):
        current_line = lines[index]
        current_indent = len(current_line) - len(current_line.lstrip())
        if index != start and current_line.strip() and current_indent <= indent:
            break

        block.append(current_line)
        index += 1

    return block, index


def _extract_description(block_text: str) -> str | None:
    match = DESCRIPTION_PATTERN.search(block_text)
    if not match:
        return None

    prefix = match.group("prefix") or ""
    quote = match.group("quote")
    value = match.group("value")
    literal = f"{prefix}{quote}{value}{quote}"

    # datamodel-code-generator emits standard string literals, but fall back to raw text on parse errors.
    try:
        parsed = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        return value.replace("\\n", "\n")

    if isinstance(parsed, str):
        return parsed
    return str(parsed)


def _inject_enum_aliases(content: str) -> str:
    enum_lines = [
        f"{name} = Literal[{', '.join(repr(value) for value in values)}]" for name, values in ENUM_LITERAL_MAP.items()
    ]
    if not enum_lines:
        return content
    block = "\n".join(enum_lines) + "\n\n"
    class_index = content.find("\nclass ")
    if class_index == -1:
        return content
    insertion_point = class_index + 1  # include leading newline
    return content[:insertion_point] + block + content[insertion_point:]


def _remove_unused_models(content: str) -> str:
    for model_name in MODELS_TO_REMOVE:
        pattern = re.compile(
            rf"^(class {model_name}\([\s\S]*?\):)([\s\S]*?)(?=^\S|\Z)",
            re.MULTILINE,
        )
        content, count = pattern.subn("", content)
        if count > 0:
            print(f"Removed unused model: {model_name}", file=sys.stderr)
    return content


if __name__ == "__main__":
    main()
