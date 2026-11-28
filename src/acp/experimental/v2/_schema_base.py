from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, get_args, get_origin

import pydantic
from pydantic import (
    ModelWrapValidatorHandler,
    ValidationInfo,
    model_validator,
)

from acp._deserialize import pop_defaulted_fields, push_defaulted_fields
from acp._schema_base import BaseModel as _BaseModel

if TYPE_CHECKING:
    from .schema import InitialBooleanConfigOption, InitialIdConfigOption, OtherInitialConfigOption

ElicitationContent = dict[str, Any]
WIRE_VALIDATION_CONTEXT: dict[str, bool] = {"require_tags": True}
_NON_ERROR_STOP_REASONS: frozenset[str] = frozenset({
    "end_turn",
    "max_tokens",
    "max_turn_requests",
    "refusal",
    "cancelled",
})


@cache
def _required_literal_fields(model_cls: type[pydantic.BaseModel]) -> tuple[tuple[str, str], ...]:
    fields: list[tuple[str, str]] = []
    for name, field in model_cls.model_fields.items():
        if get_origin(field.annotation) is Literal and len(get_args(field.annotation)) == 1:
            fields.append((name, field.alias or name))
    return tuple(fields)


class BaseModel(_BaseModel):
    """Runtime behavior shared by generated ACP v2 schema models."""

    # datamodel-code-generator does not emit the `not` constraints used by ACP's
    # open unions, so catch-all variants must reject tags owned by known variants.
    _reserved_tags: ClassVar[dict[str, tuple[str, frozenset[str]]]] = {
        "OtherAuthMethod": ("type", frozenset({"agent", "terminal"})),
        "OtherAvailableCommand": ("type", frozenset({"command", "skill"})),
        "OtherAvailableCommandInput": ("type", frozenset({"text"})),
        "OtherBackgroundWork": ("type", frozenset({"subagent", "terminal", "tool_call"})),
        "OtherContentBlock": ("type", frozenset({"audio", "image", "resource", "resource_link", "skill", "text"})),
        "OtherCursorAnchor": (
            "type",
            frozenset({"user_message", "agent_message", "agent_thought", "session_message", "tool_call", "compaction"}),
        ),
        "OtherDiffChange": ("operation", frozenset({"add", "copy", "delete", "modify", "move"})),
        "OtherInitialConfigOption": ("type", frozenset({"boolean", "id"})),
        "OtherMcpServer": ("type", frozenset({"acp", "http", "stdio"})),
        "OtherPermissionOutcome": ("outcome", frozenset({"cancelled", "selected"})),
        "OtherPermissionSubject": ("type", frozenset({"command", "tool_call"})),
        "OtherPlanUpdateContent": ("type", frozenset({"items", "markdown"})),
        "OtherReplayFrom": ("type", frozenset({"start"})),
        "OtherSessionConfigOption": ("type", frozenset({"boolean", "select"})),
        "OtherSessionStateUpdate": ("state", frozenset({"idle", "requires_action", "running", "unknown"})),
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
                "notice",
                "plan_removed",
                "plan_update",
                "cursor_update",
                "prompt_suggestions_update",
                "replay_complete",
                "session_info_update",
                "state_update",
                "terminal_output_chunk",
                "terminal_update",
                "tool_call_content_chunk",
                "tool_call_update",
                "usage_update",
                "user_message",
                "user_message_chunk",
                "subagent_update",
                "session_message",
                "session_message_chunk",
            }),
        ),
        "OtherState": ("state", frozenset({"idle", "requires_action", "running", "unknown"})),
        "OtherToolCallContent": ("type", frozenset({"content", "diff", "session", "terminal"})),
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

    @model_validator(mode="wrap")
    @classmethod
    def _discard_defaulted_fields(cls, value: Any, handler: ModelWrapValidatorHandler[Any]) -> Any:
        discarded: set[str] = set()
        token = push_defaulted_fields(discarded)
        try:
            model = handler(value)
        finally:
            pop_defaulted_fields(token)
        if discarded:
            model.model_fields_set.difference_update(discarded)
        return model

    @model_validator(mode="before")
    @classmethod
    def _require_wire_tags(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, dict) and info.context and info.context.get("require_tags"):
            for name, alias in _required_literal_fields(cls):
                if alias not in value and name not in value:
                    raise ValueError(f"Field required: {alias!r}")
        return value


class ElicitationRequestScopeModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _validate_scope(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.keys() & {"sessionId", "session_id", "toolCallId", "tool_call_id"}:
            raise ValueError("Request-scoped elicitation forbids sessionId and toolCallId")
        return value


class ElicitationSessionScopeModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _validate_scope(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.keys() & {"requestId", "request_id"}:
            raise ValueError("Session-scoped elicitation forbids requestId")
        return value


class TitledMultiSelectItemsModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _validate_omitted_type(cls, value: Any) -> Any:
        if isinstance(value, dict) and "type" in value:
            raise ValueError("Titled multi-select items must omit type")
        return value


class IdleStateUpdateModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _ignore_error_for_non_error_stop_reason(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "error" not in value:
            return value
        stop_reason = value.get("stopReason", value.get("stop_reason"))
        if stop_reason is None or (isinstance(stop_reason, str) and stop_reason in _NON_ERROR_STOP_REASONS):
            return {key: item for key, item in value.items() if key != "error"}
        return value


def validate_mcp_tool_call_id(value: str | None, info: ValidationInfo) -> str | None:
    if value is not None and info.data.get("session_id") is None:
        raise ValueError("tool_call_id requires session_id")
    return value


_Options = TypeVar("_Options", bound=Sequence[object])


def validate_select_options(options: _Options, _info: ValidationInfo) -> _Options:
    if not options:
        raise ValueError("options must contain at least one item")
    return options


def validate_initial_config_options(
    options: list[InitialBooleanConfigOption | InitialIdConfigOption | OtherInitialConfigOption],
    _info: ValidationInfo,
) -> list[InitialBooleanConfigOption | InitialIdConfigOption | OtherInitialConfigOption]:
    if len({option.config_id for option in options}) != len(options):
        raise ValueError("configOptions contains duplicate configId values")
    return options
