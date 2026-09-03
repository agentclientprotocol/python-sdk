from typing import ClassVar

from acp._schema_base import BaseModel as _BaseModel


class BaseModel(_BaseModel):
    """Runtime behavior shared by generated ACP v2 schema models."""

    # datamodel-code-generator does not emit the `not` constraints used by ACP's
    # open unions, so catch-all variants must reject tags owned by known variants.
    _reserved_tags: ClassVar[dict[str, tuple[str, frozenset[str]]]] = {
        "OtherAuthMethod": ("type", frozenset({"agent", "terminal"})),
        "OtherAvailableCommandInput": ("type", frozenset({"text"})),
        "OtherContentBlock": ("type", frozenset({"audio", "image", "resource", "resource_link", "text"})),
        "OtherDiffChange": ("operation", frozenset({"add", "copy", "delete", "modify", "move"})),
        "OtherMcpServer": ("type", frozenset({"acp", "http", "stdio"})),
        "OtherNesSuggestion": ("kind", frozenset({"edit", "jump", "rename", "searchAndReplace"})),
        "OtherPermissionOutcome": ("outcome", frozenset({"cancelled", "selected"})),
        "OtherPermissionSubject": ("type", frozenset({"command", "tool_call"})),
        "OtherPlanUpdateContent": ("type", frozenset({"file", "items", "markdown"})),
        "OtherReplayFrom": ("type", frozenset({"start"})),
        "OtherSessionConfigOption": ("type", frozenset({"boolean", "select"})),
        "OtherSessionStateUpdate": ("state", frozenset({"idle", "requires_action", "running"})),
        "OtherSessionUpdate": (
            "sessionUpdate",
            frozenset({
                "agent_message",
                "agent_message_chunk",
                "agent_thought",
                "agent_thought_chunk",
                "available_commands_update",
                "compaction_summary_chunk",
                "compaction_update",
                "config_option_update",
                "plan_removed",
                "plan_update",
                "session_info_update",
                "state_update",
                "terminal_output_chunk",
                "terminal_update",
                "tool_call_content_chunk",
                "tool_call_update",
                "usage_update",
                "user_message",
                "user_message_chunk",
            }),
        ),
        "OtherState": ("state", frozenset({"idle", "requires_action", "running"})),
        "OtherToolCallContent": ("type", frozenset({"content", "diff", "terminal"})),
        "SetSessionConfigOptionOtherRequest": ("type", frozenset({"boolean", "id"})),
        "CreateOtherSessionElicitationRequest": ("mode", frozenset({"form", "url"})),
        "CreateOtherRequestElicitationRequest": ("mode", frozenset({"form", "url"})),
        "OtherElicitationResponse": ("action", frozenset({"accept", "cancel", "decline"})),
        "ElicitationOtherPropertySchema": (
            "type",
            frozenset({"array", "boolean", "integer", "number", "string"}),
        ),
        "OtherMultiSelectItems": ("type", frozenset({"string"})),
    }
