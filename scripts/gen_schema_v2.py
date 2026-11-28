from __future__ import annotations

from typing import Any

from datamodel_code_generator.validators import ValidatorDefinition, ValidatorMode

from ._schema_semantics import (
    ROOT,
    SchemaSemantics,
    inline_model_ref,
    variant_model_map,
)

MODEL_NAME_MAP = {
    "#/$defs/AvailableCommandsUpdate": "AvailableCommandsUpdateBase",
    "#/$defs/CancelledPermissionOutcome": "CancelledPermissionOutcomeBase",
    "#/$defs/ConfigOptionUpdate": "ConfigOptionUpdateBase",
    "#/$defs/SessionInfoUpdate": "SessionInfoUpdateBase",
    "#/$defs/StringMultiSelectItems": "StringMultiSelectItemsBase",
    "#/$defs/UsageUpdate": "UsageUpdateBase",
}
for variant_names in (
    variant_model_map("AgentResponse", "anyOf", "object", ("AgentResponseMessage", "AgentErrorMessage")),
    variant_model_map("ClientResponse", "anyOf", "object", ("ClientResponseMessage", "ClientErrorMessage")),
    variant_model_map("ProxyResponse", "anyOf", "object", ("ProxyResponseMessage", "ProxyErrorMessage")),
    variant_model_map("AuthMethod", "anyOf", "allOf", ("TerminalAuthMethod", "AgentAuthMethod")),
    variant_model_map("AvailableCommand", "anyOf", "allOf", ("SlashCommand", "SkillCommand")),
    variant_model_map("AvailableCommandInput", "anyOf", "allOf", ("TextAvailableCommandInput",)),
    variant_model_map(
        "BackgroundWork",
        "anyOf",
        "allOf",
        ("TerminalBackgroundWork", "ToolCallBackgroundWork", "SubagentBackgroundWork"),
    ),
    variant_model_map(
        "InitialSessionConfigOption",
        "anyOf",
        "object",
        ("InitialIdConfigOption", "InitialBooleanConfigOption", "OtherInitialConfigOption"),
    ),
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
            "SkillContentBlock",
        ),
    ),
    variant_model_map(
        "DiffChange",
        "anyOf",
        "allOf",
        ("AddDiffChange", "DeleteDiffChange", "ModifyDiffChange", "MoveDiffChange", "CopyDiffChange"),
    ),
    variant_model_map("McpServer", "anyOf", "allOf", ("HttpMcpServer", "AcpMcpServer", "StdioMcpServer")),
    variant_model_map(
        "CursorAnchor",
        "anyOf",
        "allOf",
        (
            "UserMessageCursorAnchor",
            "AgentMessageCursorAnchor",
            "AgentThoughtCursorAnchor",
            "SessionMessageCursorAnchor",
            "ToolCallCursorAnchor",
            "CompactionCursorAnchor",
        ),
    ),
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
        ("RunningState", "IdleState", "RequiresActionState", "UnknownState"),
    ),
    variant_model_map(
        "ToolCallContent",
        "anyOf",
        "allOf",
        ("ContentToolCallContent", "DiffToolCallContent", "TerminalToolCallContent", "SessionToolCallContent"),
    ),
    variant_model_map("PlanUpdateContent", "anyOf", "allOf", ("PlanUpdateItems", "PlanUpdateMarkdown")),
    variant_model_map(
        "ElicitationFormMode", "oneOf", "allOf", ("ElicitationFormSessionMode", "ElicitationFormRequestMode")
    ),
    variant_model_map(
        "ElicitationUrlMode", "oneOf", "allOf", ("ElicitationUrlSessionMode", "ElicitationUrlRequestMode")
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
            "SessionNotice",
            "SessionCompactionUpdate",
            "SessionCompactionSummaryChunk",
            "SessionReplayComplete",
            "SessionPromptSuggestionsUpdate",
            "SessionCursorUpdate",
            "SessionSubagentUpdate",
            "SessionMessageUpdate",
            "SessionMessageChunkUpdate",
        ),
    ),
):
    MODEL_NAME_MAP.update(variant_names)

MODEL_NAME_MAP.update({
    inline_model_ref("BackgroundWork", ("anyOf", 3), ("object", None)): "OtherBackgroundWork",
    inline_model_ref("MessageMcpResponse", ("anyOf", 0), ("object", None)): "McpResultResponse",
    inline_model_ref("MessageMcpResponse", ("anyOf", 1), ("object", None)): "McpErrorResponse",
    inline_model_ref("AuthMethod", ("anyOf", 2), ("object", None)): "OtherAuthMethod",
    inline_model_ref("AvailableCommand", ("anyOf", 2), ("object", None)): "OtherAvailableCommand",
    inline_model_ref("AvailableCommandInput", ("anyOf", 1), ("object", None)): "OtherAvailableCommandInput",
    inline_model_ref("ContentBlock", ("anyOf", 6), ("object", None)): "OtherContentBlock",
    inline_model_ref("CursorAnchor", ("anyOf", 6), ("object", None)): "OtherCursorAnchor",
    inline_model_ref("DiffChange", ("anyOf", 5), ("object", None)): "OtherDiffChange",
    inline_model_ref("McpServer", ("anyOf", 3), ("object", None)): "OtherMcpServer",
    inline_model_ref("ReplayFrom", ("anyOf", 1), ("object", None)): "OtherReplayFrom",
    inline_model_ref("RequestPermissionOutcome", ("anyOf", 0), ("allOf", None)): "CancelledPermissionOutcome",
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
    inline_model_ref("StateUpdate", ("anyOf", 4), ("object", None)): "OtherState",
    inline_model_ref("ToolCallContent", ("anyOf", 4), ("object", None)): "OtherToolCallContent",
    inline_model_ref("PlanUpdateContent", ("anyOf", 2), ("object", None)): "OtherPlanUpdateContent",
    inline_model_ref("ElicitationPropertySchema", ("anyOf", 5), ("object", None)): ("ElicitationOtherPropertySchema"),
    inline_model_ref("MultiSelectItems", ("anyOf", 1), ("object", None)): "OtherMultiSelectItems",
    inline_model_ref("SessionUpdate", ("anyOf", 26), ("object", None)): "OtherSessionUpdate",
    inline_model_ref("SessionUpdate", ("anyOf", 6), ("allOf", 0), ("allOf", None)): ("RunningSessionStateUpdateBase"),
    inline_model_ref("SessionUpdate", ("anyOf", 6), ("allOf", 1), ("allOf", None)): ("IdleSessionStateUpdateBase"),
    inline_model_ref("SessionUpdate", ("anyOf", 6), ("allOf", 2), ("allOf", None)): (
        "RequiresActionSessionStateUpdateBase"
    ),
    inline_model_ref("SessionUpdate", ("anyOf", 6), ("allOf", 3), ("allOf", None)): "UnknownSessionStateUpdateBase",
    inline_model_ref("SessionUpdate", ("anyOf", 6), ("allOf", 4), ("object", None)): ("OtherSessionStateUpdateBase"),
    inline_model_ref("SessionUpdate", ("anyOf", 6), ("allOf", None), ("allOf", None)): ("SessionStateUpdateBase"),
})
for index, name in enumerate((
    "RunningSessionStateUpdate",
    "IdleSessionStateUpdate",
    "RequiresActionSessionStateUpdate",
    "UnknownSessionStateUpdate",
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
    inline_model_ref("CreateElicitationRequest", ("anyOf", 2), ("oneOf", 0), ("allOf", None)): (
        "CreateOtherSessionElicitationRequest"
    ),
    inline_model_ref("CreateElicitationRequest", ("anyOf", 2), ("oneOf", 1), ("allOf", None)): (
        "CreateOtherRequestElicitationRequest"
    ),
})


def _prepare_schema(schema: dict[str, Any]) -> None:
    definitions = schema["$defs"]
    for definition in definitions.values():
        for field in definition.get("properties", {}).values():
            if field.get("type") == "array" and field.get("x-deserialize-default-on-null"):
                field["default"] = []
    # A nested stop-reason union loses its fields when datamodel-code-generator
    # composes it into SessionUpdate. Use the common nullable string discriminator
    # and the error variant's optional details on one model.
    idle = definitions["IdleStateUpdate"]
    idle.pop("anyOf")
    idle["properties"]["error"] = definitions["ErrorStopReason"]["properties"]["error"]

    # Keep select options as a direct list union instead of numbered RootModel
    # wrappers; a field validator enforces minItems: 1 at runtime.
    for branch in definitions["SessionConfigSelectOptions"]["anyOf"]:
        branch.pop("minItems", None)

    # Inline IdempotencyKey bounds onto PromptRequest so callers pass plain str.
    idempotency_key = definitions.pop("IdempotencyKey")
    min_length = idempotency_key["minLength"]
    max_length = idempotency_key["maxLength"]
    prompt_key = definitions["PromptRequest"]["properties"]["idempotencyKey"]
    prompt_key.pop("anyOf", None)
    prompt_key.update(
        type=["string", "null"],
        minLength=min_length,
        maxLength=max_length,
    )


SEMANTICS = SchemaSemantics(
    schema_json=ROOT / "schema" / "v2" / "schema.json",
    version_file=ROOT / "schema" / "v2" / "VERSION",
    schema_out=ROOT / "src" / "acp" / "experimental" / "v2" / "schema.py",
    base_class="acp.experimental.v2._schema_base.BaseModel",
    compatibility_aliases="""\
IdempotencyKey = Annotated[str, Field(min_length=1, max_length=255)]


MessageMcpResponse = Annotated[Union[McpResultResponse, McpErrorResponse], Field(union_mode="left_to_right")]
SetSessionConfigOptionRequest = (
    SetSessionConfigOptionIdRequest | SetSessionConfigOptionBooleanRequest | SetSessionConfigOptionOtherRequest
)
CreateElicitationRequest = (
    CreateOtherSessionElicitationRequest | CreateOtherRequestElicitationRequest
    | CreateFormSessionElicitationRequest | CreateFormRequestElicitationRequest
    | CreateUrlSessionElicitationRequest | CreateUrlRequestElicitationRequest
)
CreateElicitationResponse = (
    AcceptElicitationResponse | DeclineElicitationResponse | CancelElicitationResponse | OtherElicitationResponse
)
ProxySuccessorResponse = Any
ContentBlock = (
    TextContentBlock
    | ImageContentBlock
    | AudioContentBlock
    | ResourceContentBlock
    | EmbeddedResourceContentBlock
    | SkillContentBlock
    | OtherContentBlock
)
AvailableCommand = SlashCommand | SkillCommand | OtherAvailableCommand
BackgroundWork = TerminalBackgroundWork | ToolCallBackgroundWork | SubagentBackgroundWork | OtherBackgroundWork
StateUpdate = RunningState | IdleState | RequiresActionState | UnknownState | OtherState
SessionUpdate = (
    UserMessageChunk
    | UserMessageUpdate
    | AgentMessageChunk
    | AgentMessageUpdate
    | AgentThoughtChunk
    | AgentThoughtUpdate
    | ToolCallContentChunkUpdate
    | SessionToolCallUpdate
    | SessionTerminalUpdate
    | SessionTerminalOutputChunk
    | SessionPlanUpdate
    | SessionPlanRemovedUpdate
    | AvailableCommandsUpdate
    | ConfigOptionUpdate
    | SessionInfoUpdate
    | UsageUpdate
    | SessionNotice
    | SessionCompactionUpdate
    | SessionCompactionSummaryChunk
    | SessionReplayComplete
    | SessionPromptSuggestionsUpdate
    | SessionCursorUpdate
    | SessionSubagentUpdate
    | SessionMessageUpdate
    | SessionMessageChunkUpdate
    | OtherSessionUpdate
    | RunningSessionStateUpdate
    | IdleSessionStateUpdate
    | RequiresActionSessionStateUpdate
    | UnknownSessionStateUpdate
    | OtherSessionStateUpdate
)
""",
    model_name_map=MODEL_NAME_MAP,
    validate_renamed_models=True,
    coerce_legacy_protocol_version=False,
    prepare_schema=_prepare_schema,
    base_class_map={
        "IdleStateUpdate": "acp.experimental.v2._schema_base.IdleStateUpdateModel",
        "ElicitationRequestScope": "acp.experimental.v2._schema_base.ElicitationRequestScopeModel",
        "ElicitationSessionScope": "acp.experimental.v2._schema_base.ElicitationSessionScopeModel",
        "TitledMultiSelectItems": "acp.experimental.v2._schema_base.TitledMultiSelectItemsModel",
    },
    extra_validators={
        "MessageMcpRequest": [
            ValidatorDefinition(
                field="tool_call_id",
                function="acp.experimental.v2._schema_base.validate_mcp_tool_call_id",
                mode=ValidatorMode.AFTER,
            )
        ],
        **{
            model: [
                ValidatorDefinition(
                    field="options",
                    function="acp.experimental.v2._schema_base.validate_select_options",
                    mode=ValidatorMode.AFTER,
                )
            ]
            for model in ("SessionConfigSelect", "SessionConfigSelectGroup")
        },
        **{
            model: [
                ValidatorDefinition(
                    field="config_options",
                    function="acp.experimental.v2._schema_base.validate_initial_config_options",
                    mode=ValidatorMode.AFTER,
                )
            ]
            for model in ("NewSessionRequest", "ForkSessionRequest", "ResumeSessionRequest")
        },
    },
    field_type_overrides={
        "ElicitationAcceptAction.content": "acp.experimental.v2._schema_base.ElicitationContent",
        "Error.code": "pydantic.StrictInt",
        "InitializeRequest.protocol_version": "pydantic.StrictInt",
        "InitializeResponse.protocol_version": "pydantic.StrictInt",
        "Error.data": "typing.Any",
        "McpError.code": "pydantic.StrictInt",
        # A successor's result stays opaque until the originating method is known.
        "ProxyResponseMessage.result": "typing.Any",
        "InitialBooleanConfigOption.value": "pydantic.StrictBool",
        "SetSessionConfigOptionBooleanRequest.value": "pydantic.StrictBool",
    },
)
