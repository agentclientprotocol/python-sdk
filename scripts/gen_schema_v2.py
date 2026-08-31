from __future__ import annotations

from ._schema_semantics import (
    ROOT,
    SchemaSemantics,
    inline_model_ref,
    variant_model_map,
)

MODEL_NAME_MAP = {
    "#/$defs/AvailableCommandsUpdate": "AvailableCommandsUpdateBase",
    "#/$defs/ConfigOptionUpdate": "ConfigOptionUpdateBase",
    "#/$defs/SessionInfoUpdate": "SessionInfoUpdateBase",
    "#/$defs/StringMultiSelectItems": "StringMultiSelectItemsBase",
    "#/$defs/UsageUpdate": "UsageUpdateBase",
}
for variant_names in (
    variant_model_map("AgentResponse", "anyOf", "object", ("AgentResponseMessage", "AgentErrorMessage")),
    variant_model_map("ClientResponse", "anyOf", "object", ("ClientResponseMessage", "ClientErrorMessage")),
    variant_model_map("AuthMethod", "anyOf", "allOf", ("TerminalAuthMethod", "AgentAuthMethod")),
    variant_model_map("AvailableCommandInput", "anyOf", "allOf", ("TextAvailableCommandInput",)),
    variant_model_map(
        "ContentBlock",
        "anyOf",
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
        "DiffChange",
        "anyOf",
        "allOf",
        ("AddDiffChange", "DeleteDiffChange", "ModifyDiffChange", "MoveDiffChange", "CopyDiffChange"),
    ),
    variant_model_map("McpServer", "anyOf", "allOf", ("HttpMcpServer", "AcpMcpServer", "StdioMcpServer")),
    variant_model_map("ReplayFrom", "anyOf", "allOf", ("ReplayFromStartVariant",)),
    variant_model_map(
        "RequestPermissionSubject",
        "anyOf",
        "allOf",
        ("ToolCallPermissionSubjectVariant", "CommandPermissionSubjectVariant"),
    ),
    variant_model_map(
        "SessionConfigOption",
        "anyOf",
        "allOf",
        ("SelectSessionConfigOption", "BooleanSessionConfigOption"),
    ),
    variant_model_map(
        "StateUpdate",
        "anyOf",
        "allOf",
        ("RunningState", "IdleState", "RequiresActionState"),
    ),
    variant_model_map(
        "ToolCallContent",
        "anyOf",
        "allOf",
        ("ContentToolCallContent", "DiffToolCallContent", "TerminalToolCallContent"),
    ),
    variant_model_map(
        "PlanUpdateContent",
        "anyOf",
        "allOf",
        ("PlanUpdateItems", "PlanUpdateFile", "PlanUpdateMarkdown"),
    ),
    variant_model_map(
        "NesSuggestion",
        "anyOf",
        "allOf",
        (
            "NesEditSuggestionVariant",
            "NesJumpSuggestionVariant",
            "NesRenameSuggestionVariant",
            "NesSearchAndReplaceSuggestionVariant",
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
    variant_model_map("MultiSelectItems", "anyOf", "allOf", ("StringMultiSelectItems",)),
    variant_model_map(
        "SessionUpdate",
        "anyOf",
        "allOf",
        (
            "UserMessageChunk",
            "UserMessageUpdate",
            "AgentMessageChunk",
            "AgentMessageUpdate",
            "AgentThoughtChunk",
            "AgentThoughtUpdate",
            "SessionStateUpdate",
            "ToolCallContentChunkUpdate",
            "SessionToolCallUpdate",
            "SessionTerminalUpdate",
            "SessionTerminalOutputChunk",
            "SessionPlanUpdate",
            "SessionPlanRemovedUpdate",
            "AvailableCommandsUpdate",
            "ConfigOptionUpdate",
            "SessionInfoUpdate",
            "UsageUpdate",
            "SessionCompactionUpdate",
            "SessionCompactionSummaryChunk",
        ),
    ),
):
    MODEL_NAME_MAP.update(variant_names)

MODEL_NAME_MAP.update({
    inline_model_ref("AuthMethod", ("anyOf", 2), ("object", None)): "OtherAuthMethod",
    inline_model_ref("AvailableCommandInput", ("anyOf", 1), ("object", None)): "OtherAvailableCommandInput",
    inline_model_ref("ContentBlock", ("anyOf", 5), ("object", None)): "OtherContentBlock",
    inline_model_ref("DiffChange", ("anyOf", 5), ("object", None)): "OtherDiffChange",
    inline_model_ref("McpServer", ("anyOf", 3), ("object", None)): "OtherMcpServer",
    inline_model_ref("ReplayFrom", ("anyOf", 1), ("object", None)): "OtherReplayFrom",
    inline_model_ref("RequestPermissionOutcome", ("anyOf", 0), ("object", None)): ("CancelledPermissionOutcome"),
    inline_model_ref("RequestPermissionOutcome", ("anyOf", 1), ("allOf", None)): ("SelectedPermissionOutcomeVariant"),
    inline_model_ref("RequestPermissionOutcome", ("anyOf", 2), ("object", None)): "OtherPermissionOutcome",
    inline_model_ref("RequestPermissionSubject", ("anyOf", 2), ("object", None)): "OtherPermissionSubject",
    inline_model_ref("SessionConfigOption", ("anyOf", 2), ("object", None)): "OtherSessionConfigOption",
    inline_model_ref("SetSessionConfigOptionRequest", ("anyOf", 0), ("object", None)): (
        "SetSessionConfigOptionIdRequest"
    ),
    inline_model_ref("SetSessionConfigOptionRequest", ("anyOf", 1), ("object", None)): (
        "SetSessionConfigOptionBooleanRequest"
    ),
    inline_model_ref("SetSessionConfigOptionRequest", ("anyOf", 2), ("object", None)): (
        "SetSessionConfigOptionOtherRequest"
    ),
    inline_model_ref("StateUpdate", ("anyOf", 3), ("object", None)): "OtherState",
    inline_model_ref("ToolCallContent", ("anyOf", 3), ("object", None)): "OtherToolCallContent",
    inline_model_ref("PlanUpdateContent", ("anyOf", 3), ("object", None)): "OtherPlanUpdateContent",
    inline_model_ref("NesSuggestion", ("anyOf", 4), ("object", None)): "OtherNesSuggestion",
    inline_model_ref("ElicitationPropertySchema", ("anyOf", 5), ("object", None)): ("ElicitationOtherPropertySchema"),
    inline_model_ref("MultiSelectItems", ("anyOf", 1), ("object", None)): "OtherMultiSelectItems",
    inline_model_ref("SessionUpdate", ("anyOf", 19), ("object", None)): "OtherSessionUpdate",
    inline_model_ref("SessionUpdate", ("anyOf", 6), ("allOf", 0), ("allOf", None)): ("RunningSessionStateUpdateBase"),
    inline_model_ref("SessionUpdate", ("anyOf", 6), ("allOf", 1), ("allOf", None)): ("IdleSessionStateUpdateBase"),
    inline_model_ref("SessionUpdate", ("anyOf", 6), ("allOf", 2), ("allOf", None)): (
        "RequiresActionSessionStateUpdateBase"
    ),
    inline_model_ref("SessionUpdate", ("anyOf", 6), ("allOf", 3), ("object", None)): ("OtherSessionStateUpdateBase"),
    inline_model_ref("SessionUpdate", ("anyOf", 6), ("allOf", None), ("allOf", None)): ("SessionStateUpdateBase"),
})
for index, name in enumerate((
    "RunningSessionStateUpdate",
    "IdleSessionStateUpdate",
    "RequiresActionSessionStateUpdate",
    "OtherSessionStateUpdate",
)):
    MODEL_NAME_MAP[inline_model_ref("SessionUpdate", ("anyOf", 6), ("allOf", None), (f"union_model-{index}", None))] = (
        name
    )

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

SEMANTICS = SchemaSemantics(
    schema_json=ROOT / "schema" / "v2" / "schema.json",
    version_file=ROOT / "schema" / "v2" / "VERSION",
    schema_out=ROOT / "src" / "acp" / "experimental" / "v2" / "schema.py",
    model_name_map=MODEL_NAME_MAP,
)
