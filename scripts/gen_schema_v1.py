from __future__ import annotations

import textwrap

from ._schema_semantics import (
    ROOT,
    SchemaSemantics,
    inline_model_ref,
    variant_model_map,
)

MODEL_NAME_MAP = {
    "#/$defs/AvailableCommandsUpdate": "AvailableCommandsUpdateBase",
    "#/$defs/ConfigOptionUpdate": "ConfigOptionUpdateBase",
    "#/$defs/CurrentModeUpdate": "CurrentModeUpdateBase",
    "#/$defs/SessionInfoUpdate": "SessionInfoUpdateBase",
    "#/$defs/StringMultiSelectItems": "StringMultiSelectItemsBase",
    "#/$defs/UsageUpdate": "UsageUpdateBase",
}
for variant_names in (
    variant_model_map("AgentResponse", "anyOf", "object", ("AgentResponseMessage", "AgentErrorMessage")),
    variant_model_map("ClientResponse", "anyOf", "object", ("ClientResponseMessage", "ClientErrorMessage")),
    variant_model_map("AuthMethod", "anyOf", "allOf", ("EnvVarAuthMethod", "TerminalAuthMethod")),
    variant_model_map("McpServer", "anyOf", "allOf", ("HttpMcpServer", "SseMcpServer", "AcpMcpServer")),
    variant_model_map(
        "SetSessionConfigOptionRequest",
        "anyOf",
        "object",
        ("SetSessionConfigOptionBooleanRequest", "SetSessionConfigOptionSelectRequest"),
    ),
    variant_model_map(
        "ContentBlock",
        "oneOf",
        "allOf",
        (
            "TextContentBlock",
            "ImageContentBlock",
            "AudioContentBlock",
            "ResourceContentBlock",
            "EmbeddedResourceContentBlock",
        ),
    ),
    variant_model_map(
        "ToolCallContent",
        "oneOf",
        "allOf",
        ("ContentToolCallContent", "FileEditToolCallContent", "TerminalToolCallContent"),
    ),
    variant_model_map(
        "PlanUpdateContent",
        "oneOf",
        "allOf",
        ("PlanUpdateItems", "PlanUpdateFile", "PlanUpdateMarkdown"),
    ),
    variant_model_map(
        "NesSuggestion",
        "oneOf",
        "allOf",
        (
            "NesEditSuggestionVariant",
            "NesJumpSuggestionVariant",
            "NesRenameSuggestionVariant",
            "NesSearchAndReplaceSuggestionVariant",
        ),
    ),
    variant_model_map(
        "SessionUpdate",
        "oneOf",
        "allOf",
        (
            "UserMessageChunk",
            "AgentMessageChunk",
            "AgentThoughtChunk",
            "ToolCallStart",
            "ToolCallProgress",
            "AgentPlanUpdate",
            "AgentPlanContentUpdate",
            "AgentPlanRemovedUpdate",
            "AvailableCommandsUpdate",
            "CurrentModeUpdate",
            "ConfigOptionUpdate",
            "SessionInfoUpdate",
            "UsageUpdate",
        ),
    ),
    variant_model_map(
        "ElicitationFormMode",
        "anyOf",
        "allOf",
        ("ElicitationFormSessionMode", "ElicitationFormRequestMode"),
    ),
    variant_model_map(
        "ElicitationUrlMode",
        "anyOf",
        "allOf",
        ("ElicitationUrlSessionMode", "ElicitationUrlRequestMode"),
    ),
    variant_model_map(
        "ElicitationPropertySchema",
        "anyOf",
        "allOf",
        (
            "ElicitationStringPropertySchema",
            "ElicitationNumberPropertySchema",
            "ElicitationIntegerPropertySchema",
            "ElicitationBooleanPropertySchema",
            "ElicitationMultiSelectPropertySchema",
        ),
    ),
):
    MODEL_NAME_MAP.update(variant_names)

MODEL_NAME_MAP.update({
    inline_model_ref("RequestPermissionOutcome", ("oneOf", 0), ("object", None)): "DeniedOutcome",
    inline_model_ref("RequestPermissionOutcome", ("oneOf", 1), ("allOf", None)): "AllowedOutcome",
    inline_model_ref("ElicitationPropertySchema", ("anyOf", 5), ("object", None)): ("ElicitationOtherPropertySchema"),
    inline_model_ref("MultiSelectItems", ("anyOf", 0), ("allOf", None)): "StringMultiSelectItems",
    inline_model_ref("MultiSelectItems", ("anyOf", 1), ("object", None)): "OtherMultiSelectItems",
})
MODEL_NAME_MAP.update({
    inline_model_ref("CreateElicitationResponse", ("anyOf", 0), ("allOf", None)): "AcceptElicitationResponse",
    inline_model_ref("CreateElicitationResponse", ("anyOf", 1), ("object", None)): "DeclineElicitationResponse",
    inline_model_ref("CreateElicitationResponse", ("anyOf", 2), ("object", None)): "CancelElicitationResponse",
    inline_model_ref("CreateElicitationResponse", ("anyOf", 3), ("object", None)): "OtherElicitationResponse",
    inline_model_ref("CreateElicitationRequest", ("anyOf", 0), ("allOf", None), ("allOf", None)): (
        "CreateFormElicitationRequestBase"
    ),
    inline_model_ref("CreateElicitationRequest", ("anyOf", 0), ("allOf", 0), ("allOf", None)): (
        "CreateFormSessionElicitationRequestBase"
    ),
    inline_model_ref("CreateElicitationRequest", ("anyOf", 0), ("allOf", 1), ("allOf", None)): (
        "CreateFormRequestElicitationRequestBase"
    ),
    inline_model_ref("CreateElicitationRequest", ("anyOf", 0), ("allOf", None), ("union_model-0", None)): (
        "CreateFormSessionElicitationRequest"
    ),
    inline_model_ref("CreateElicitationRequest", ("anyOf", 0), ("allOf", None), ("union_model-1", None)): (
        "CreateFormRequestElicitationRequest"
    ),
    inline_model_ref("CreateElicitationRequest", ("anyOf", 1), ("allOf", None), ("allOf", None)): (
        "CreateUrlElicitationRequestBase"
    ),
    inline_model_ref("CreateElicitationRequest", ("anyOf", 1), ("allOf", 0), ("allOf", None)): (
        "CreateUrlSessionElicitationRequestBase"
    ),
    inline_model_ref("CreateElicitationRequest", ("anyOf", 1), ("allOf", 1), ("allOf", None)): (
        "CreateUrlRequestElicitationRequestBase"
    ),
    inline_model_ref("CreateElicitationRequest", ("anyOf", 1), ("allOf", None), ("union_model-0", None)): (
        "CreateUrlSessionElicitationRequest"
    ),
    inline_model_ref("CreateElicitationRequest", ("anyOf", 1), ("allOf", None), ("union_model-1", None)): (
        "CreateUrlRequestElicitationRequest"
    ),
    inline_model_ref("CreateElicitationRequest", ("anyOf", 2), ("anyOf", 0), ("allOf", None)): (
        "CreateOtherSessionElicitationRequest"
    ),
    inline_model_ref("CreateElicitationRequest", ("anyOf", 2), ("anyOf", 1), ("allOf", None)): (
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

    _AvailableCommandsUpdate = AvailableCommandsUpdateBase
    _CurrentModeUpdate = CurrentModeUpdateBase
    _ConfigOptionUpdate = ConfigOptionUpdateBase
    _SessionInfoUpdate = SessionInfoUpdateBase
    _UsageUpdate = UsageUpdateBase
    _StringMultiSelectItems = StringMultiSelectItemsBase

    class Jsonrpc(Enum):
        field_2_0 = "2.0"
    """).strip()

SEMANTICS = SchemaSemantics(
    schema_json=ROOT / "schema" / "schema.json",
    version_file=ROOT / "schema" / "VERSION",
    schema_out=ROOT / "src" / "acp" / "schema.py",
    model_name_map=MODEL_NAME_MAP,
    compatibility_aliases=COMPATIBILITY_ALIASES,
)
