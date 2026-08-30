#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import difflib
import json
import subprocess
import sys
import textwrap
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
SCHEMA_JSON = ROOT / "schema" / "schema.json"
VERSION_FILE = ROOT / "schema" / "VERSION"
SCHEMA_OUT = ROOT / "src" / "acp" / "schema.py"

UNSIGNED_TYPE_MAPPINGS = (
    "integer+uint16=integer",
    "integer+uint32=integer",
    "integer+uint64=integer",
)

OPEN_UNIONS = {
    "CreateElicitationRequest": 2,
    "CreateElicitationResponse": 3,
    "ElicitationPropertySchema": 5,
    "MultiSelectItems": 1,
}


def _inline_model_ref(definition: str, *steps: tuple[str, int | None]) -> str:
    ref = f"#/$defs/{definition}"
    for keyword, index in steps:
        ref += f"#-datamodel-code-generator-#-{keyword}-#-special-#"
        if index is not None:
            ref += f"/{index}"
    return ref


# A few schema definitions share their name with a tagged SessionUpdate or
# MultiSelectItems variant. Give the definition an internal base name so the public
# tagged model can keep the established SDK name without shadowing a class.
MODEL_NAME_MAP = {
    f"#/$defs/{name}": f"{name}Base"
    for name in (
        "AvailableCommandsUpdate",
        "ConfigOptionUpdate",
        "CurrentModeUpdate",
        "SessionInfoUpdate",
        "StringMultiSelectItems",
        "UsageUpdate",
    )
}
MODEL_NAME_MAP.update({
    _inline_model_ref("AgentResponse", ("anyOf", 0), ("object", None)): "AgentResponseMessage",
    _inline_model_ref("AgentResponse", ("anyOf", 1), ("object", None)): "AgentErrorMessage",
    _inline_model_ref("ClientResponse", ("anyOf", 0), ("object", None)): "ClientResponseMessage",
    _inline_model_ref("ClientResponse", ("anyOf", 1), ("object", None)): "ClientErrorMessage",
    _inline_model_ref("SetSessionConfigOptionRequest", ("anyOf", 0), ("object", None)): (
        "SetSessionConfigOptionBooleanRequest"
    ),
    _inline_model_ref("SetSessionConfigOptionRequest", ("anyOf", 1), ("object", None)): (
        "SetSessionConfigOptionSelectRequest"
    ),
    _inline_model_ref("CreateElicitationResponse", ("anyOf", 0), ("allOf", None)): "AcceptElicitationResponse",
    _inline_model_ref("CreateElicitationResponse", ("anyOf", 1), ("object", None)): "DeclineElicitationResponse",
    _inline_model_ref("CreateElicitationResponse", ("anyOf", 2), ("object", None)): "CancelElicitationResponse",
    _inline_model_ref("CreateElicitationResponse", ("anyOf", 3), ("object", None)): "OtherElicitationResponse",
    _inline_model_ref("ElicitationFormMode", ("anyOf", 0), ("allOf", None)): "ElicitationFormSessionMode",
    _inline_model_ref("ElicitationFormMode", ("anyOf", 1), ("allOf", None)): "ElicitationFormRequestMode",
    _inline_model_ref("ElicitationUrlMode", ("anyOf", 0), ("allOf", None)): "ElicitationUrlSessionMode",
    _inline_model_ref("ElicitationUrlMode", ("anyOf", 1), ("allOf", None)): "ElicitationUrlRequestMode",
    _inline_model_ref("ElicitationPropertySchema", ("anyOf", 0), ("allOf", None)): ("ElicitationStringPropertySchema"),
    _inline_model_ref("ElicitationPropertySchema", ("anyOf", 1), ("allOf", None)): ("ElicitationNumberPropertySchema"),
    _inline_model_ref("ElicitationPropertySchema", ("anyOf", 2), ("allOf", None)): ("ElicitationIntegerPropertySchema"),
    _inline_model_ref("ElicitationPropertySchema", ("anyOf", 3), ("allOf", None)): ("ElicitationBooleanPropertySchema"),
    _inline_model_ref("ElicitationPropertySchema", ("anyOf", 4), ("allOf", None)): (
        "ElicitationMultiSelectPropertySchema"
    ),
    _inline_model_ref("ElicitationPropertySchema", ("anyOf", 5), ("object", None)): ("ElicitationOtherPropertySchema"),
    _inline_model_ref("MultiSelectItems", ("anyOf", 0), ("allOf", None)): "StringMultiSelectItems",
    _inline_model_ref("MultiSelectItems", ("anyOf", 1), ("object", None)): "OtherMultiSelectItems",
    _inline_model_ref("CreateElicitationRequest", ("anyOf", 0), ("allOf", None), ("allOf", None)): (
        "CreateFormElicitationRequestBase"
    ),
    _inline_model_ref("CreateElicitationRequest", ("anyOf", 0), ("allOf", 0), ("allOf", None)): (
        "CreateFormSessionElicitationRequestBase"
    ),
    _inline_model_ref("CreateElicitationRequest", ("anyOf", 0), ("allOf", 1), ("allOf", None)): (
        "CreateFormRequestElicitationRequestBase"
    ),
    _inline_model_ref("CreateElicitationRequest", ("anyOf", 0), ("allOf", None), ("union_model-0", None)): (
        "CreateFormSessionElicitationRequest"
    ),
    _inline_model_ref("CreateElicitationRequest", ("anyOf", 0), ("allOf", None), ("union_model-1", None)): (
        "CreateFormRequestElicitationRequest"
    ),
    _inline_model_ref("CreateElicitationRequest", ("anyOf", 1), ("allOf", None), ("allOf", None)): (
        "CreateUrlElicitationRequestBase"
    ),
    _inline_model_ref("CreateElicitationRequest", ("anyOf", 1), ("allOf", 0), ("allOf", None)): (
        "CreateUrlSessionElicitationRequestBase"
    ),
    _inline_model_ref("CreateElicitationRequest", ("anyOf", 1), ("allOf", 1), ("allOf", None)): (
        "CreateUrlRequestElicitationRequestBase"
    ),
    _inline_model_ref("CreateElicitationRequest", ("anyOf", 1), ("allOf", None), ("union_model-0", None)): (
        "CreateUrlSessionElicitationRequest"
    ),
    _inline_model_ref("CreateElicitationRequest", ("anyOf", 1), ("allOf", None), ("union_model-1", None)): (
        "CreateUrlRequestElicitationRequest"
    ),
    _inline_model_ref("CreateElicitationRequest", ("anyOf", 2), ("anyOf", 0), ("allOf", None)): (
        "CreateOtherSessionElicitationRequest"
    ),
    _inline_model_ref("CreateElicitationRequest", ("anyOf", 2), ("anyOf", 1), ("allOf", None)): (
        "CreateOtherRequestElicitationRequest"
    ),
})

# datamodel-code-generator owns schema interpretation and its internal model names.
# This block only preserves the Python names already published by the SDK.
COMPATIBILITY_ALIASES = textwrap.dedent("""
    PermissionOptionKind = Literal["allow_once", "allow_always", "reject_once", "reject_always"]
    PlanEntryPriority = Literal["high", "medium", "low"]
    PlanEntryStatus = Literal["pending", "in_progress", "completed"]
    StopReason = Literal["end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"]
    ToolCallStatus = Literal["pending", "in_progress", "completed", "failed"]
    ToolKind = Literal[
        "read",
        "edit",
        "delete",
        "move",
        "search",
        "execute",
        "think",
        "fetch",
        "switch_mode",
        "other",
    ]

    TextContentBlock = ContentBlockText
    ImageContentBlock = ContentBlockImage
    AudioContentBlock = ContentBlockAudio
    ResourceContentBlock = ContentBlockResourceLink
    EmbeddedResourceContentBlock = ContentBlockResource

    HttpMcpServer = McpServerHttpModel
    SseMcpServer = McpServerSseModel
    AcpMcpServer = McpServerAcpModel
    DeniedOutcome = RequestPermissionOutcomeCancelled
    AllowedOutcome = RequestPermissionOutcomeSelected
    EnvVarAuthMethod = AuthMethodEnvVarModel
    TerminalAuthMethod = AuthMethodTerminalModel
    UserMessageChunk = SessionUpdateUserMessageChunk
    AgentMessageChunk = SessionUpdateAgentMessageChunk
    AgentThoughtChunk = SessionUpdateAgentThoughtChunk
    ToolCallStart = SessionUpdateToolCall
    ToolCallProgress = SessionUpdateToolCallUpdate
    AgentPlanUpdate = SessionUpdatePlan
    AgentPlanContentUpdate = SessionUpdatePlanUpdate
    AgentPlanRemovedUpdate = SessionUpdatePlanRemoved

    _AvailableCommandsUpdate = AvailableCommandsUpdateBase
    _CurrentModeUpdate = CurrentModeUpdateBase
    _ConfigOptionUpdate = ConfigOptionUpdateBase
    _SessionInfoUpdate = SessionInfoUpdateBase
    _UsageUpdate = UsageUpdateBase
    AvailableCommandsUpdate = SessionUpdateAvailableCommandsUpdate
    CurrentModeUpdate = SessionUpdateCurrentModeUpdate
    ConfigOptionUpdate = SessionUpdateConfigOptionUpdate
    SessionInfoUpdate = SessionUpdateSessionInfoUpdate
    UsageUpdate = SessionUpdateUsageUpdate

    PlanUpdateItems = PlanUpdateContentItems
    PlanUpdateFile = PlanUpdateContentFile
    PlanUpdateMarkdown = PlanUpdateContentMarkdown
    ContentToolCallContent = ToolCallContentContent
    FileEditToolCallContent = ToolCallContentDiff
    TerminalToolCallContent = ToolCallContentTerminal

    CreateOtherElicitationRequest = Union[
        CreateOtherSessionElicitationRequest,
        CreateOtherRequestElicitationRequest,
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
    ElicitationMode = Union[
        ElicitationFormSessionMode,
        ElicitationFormRequestMode,
        ElicitationUrlSessionMode,
        ElicitationUrlRequestMode,
    ]

    _StringMultiSelectItems = StringMultiSelectItemsBase

    NesEditSuggestionVariant = NesSuggestionEdit
    NesJumpSuggestionVariant = NesSuggestionJump
    NesRenameSuggestionVariant = NesSuggestionRename
    NesSearchAndReplaceSuggestionVariant = NesSuggestionSearchAndReplace

    class Jsonrpc(Enum):
        field_2_0 = "2.0"
    """).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ACP v1 schema bindings.")
    parser.add_argument("--check", action="store_true", help="Fail if the committed bindings are stale.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not generate_schema(check=args.check):
        raise SystemExit(1)


def generate_schema(*, check: bool = False) -> bool:
    candidate = render_schema()
    current = SCHEMA_OUT.read_text(encoding="utf-8") if SCHEMA_OUT.exists() else ""
    if check:
        if current == candidate:
            return True
        print(
            "".join(
                difflib.unified_diff(
                    current.splitlines(keepends=True),
                    candidate.splitlines(keepends=True),
                    fromfile=str(SCHEMA_OUT.relative_to(ROOT)),
                    tofile=f"{SCHEMA_OUT.relative_to(ROOT)} (generated)",
                )
            ),
            end="",
        )
        return False
    SCHEMA_OUT.write_text(candidate, encoding="utf-8")
    return True


def render_schema() -> str:
    """Generate v1 directly from the currently pinned JSON Schema."""
    if not SCHEMA_JSON.exists():
        raise FileNotFoundError("schema/schema.json is missing; fetch a pinned schema release first")

    schema = _schema_for_codegen(json.loads(SCHEMA_JSON.read_text(encoding="utf-8")))
    generated = generate(
        schema,
        input_file_type=InputFileType.JsonSchema,
        custom_file_header=_build_header(),
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
        model_name_map=MODEL_NAME_MAP,
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
    return _format_python(f"{generated.rstrip()}\n\n\n{COMPATIBILITY_ALIASES}\n")


def _schema_for_codegen(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop open-union constraints that Pydantic cannot represent statically."""
    patched = copy.deepcopy(schema)
    for name, catchall_index in OPEN_UNIONS.items():
        try:
            del patched["$defs"][name]["discriminator"]
            del patched["$defs"][name]["anyOf"][catchall_index]["not"]
        except KeyError:
            raise ValueError(f"{name} no longer has the expected open-union shape") from None
    return patched


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


def _build_header() -> str:
    lines = ["# Generated from schema/schema.json. Do not edit by hand."]
    if VERSION_FILE.exists() and (ref := VERSION_FILE.read_text(encoding="utf-8").strip()):
        lines.append(f"# Schema ref: {ref}")
    return "\n".join(lines)


def _format_python(source: str) -> str:
    commands = (
        ("check", "--fix"),
        ("format",),
    )
    for arguments in commands:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "ruff", *arguments, "--stdin-filename", str(SCHEMA_OUT), "-"],
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
