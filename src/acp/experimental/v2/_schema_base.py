from __future__ import annotations

from functools import cache
from typing import Any, ClassVar

import pydantic
from pydantic import (
    ModelWrapValidatorHandler,
    TypeAdapter,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_snake

from acp._deserialize import pop_defaulted_fields, push_defaulted_fields
from acp._schema_base import BaseModel as _BaseModel

from .meta import METHOD_SCHEMAS

ElicitationContent = dict[str, Any]
_NON_ERROR_STOP_REASONS = frozenset({None, "end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"})


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
        if discarded and isinstance(model, pydantic.BaseModel):
            model.model_fields_set.difference_update(discarded)
        return model

    @model_validator(mode="before")
    @classmethod
    def _validate_replay_from_tag(cls, value: Any) -> Any:
        if (
            cls.__name__ == "ResumeSessionRequest"
            and isinstance(value, dict)
            and isinstance((replay_from := value.get("replayFrom", value.get("replay_from"))), dict)
            and "type" not in replay_from
        ):
            raise ValueError("replayFrom requires an explicit 'type' field")
        return value

    @model_validator(mode="before")
    @classmethod
    def _validate_elicitation_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if cls.__name__ == "TitledMultiSelectItems" and "type" in value:
            raise ValueError("Titled multi-select items must omit type")
        scopes = {base.__name__ for base in cls.__mro__}
        if "ElicitationRequestScope" in scopes:
            forbidden = ("sessionId", "toolCallId")
        elif "ElicitationSessionScope" in scopes:
            forbidden = ("requestId",)
        else:
            return value
        if any(key in value or to_snake(key) in value for key in forbidden):
            raise ValueError("Elicitation must select one scope")
        return value

    @model_validator(mode="after")
    def _validate_mcp_attribution(self) -> BaseModel:
        if (
            type(self).__name__ == "MessageMcpRequest"
            and self.__dict__.get("tool_call_id") is not None
            and self.__dict__.get("session_id") is None
        ):
            raise ValueError("tool_call_id requires session_id")
        return self


class ErrorModel(BaseModel):
    @model_validator(mode="after")
    def _validate_version_error_data(self) -> ErrorModel:
        if self.__dict__.get("code") == -32022:
            from .schema import UnsupportedProtocolVersionErrorData

            self.__dict__["data"] = UnsupportedProtocolVersionErrorData.model_validate(self.__dict__.get("data"))
        return self


class IdleStateUpdateModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _ignore_error_for_non_error_stop_reason(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "error" not in value:
            return value
        stop_reason = value.get("stopReason", value.get("stop_reason"))
        if stop_reason in _NON_ERROR_STOP_REASONS:
            return {key: item for key, item in value.items() if key != "error"}
        return value


class SessionConfigSelectModel(BaseModel):
    @model_validator(mode="after")
    def _validate_non_empty_options(self) -> SessionConfigSelectModel:
        if not self.__dict__.get("options"):
            raise ValueError("options must contain at least one item")
        return self


@cache
def _payload_adapter(model_name: str) -> TypeAdapter[Any]:
    from . import schema

    return TypeAdapter(getattr(schema, model_name))


class MethodParamsModel(BaseModel):
    # Envelope names identify the sender. The schema registry is keyed by receiver.
    _payload_route: ClassVar[dict[str, tuple[str, str]]] = {
        "AgentNotification": ("client", "notification"),
        "AgentRequest": ("client", "request"),
        "ClientNotification": ("agent", "notification"),
        "ClientRequest": ("agent", "request"),
        "ProxyNotification": ("proxy", "notification"),
        "ProxyRequest": ("proxy", "request"),
    }

    @classmethod
    def _resolve_method_payload(cls, method: str, value: Any) -> Any:
        side, kind = next(
            cls._payload_route[base.__name__] for base in cls.__mro__ if base.__name__ in cls._payload_route
        )
        method_spec = METHOD_SCHEMAS[side].get(method)
        if method_spec is None and kind == "notification" and side != "proxy":
            method_spec = METHOD_SCHEMAS.get("protocol", {}).get(method)
        if method_spec is not None:
            model_name = method_spec.get(kind)
            if model_name is None:
                raise ValueError(f"{method!r} is not a valid {side} {kind}")
            adapter = _payload_adapter(model_name)
            if value is None:
                adapter.validate_python({})
                return None
            return adapter.validate_python(value)
        if side != "proxy" and method.startswith("_"):
            return value
        raise ValueError(f"Unknown {side} {kind} method: {method!r}")

    @field_validator("params", mode="wrap", check_fields=False)
    @classmethod
    def _validate_method_params(cls, value: Any, _handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        method = info.data.get("method") if info.data is not None else None
        if not isinstance(method, str):
            return value
        return cls._resolve_method_payload(method, value)

    @model_validator(mode="after")
    def _validate_omitted_params(self) -> MethodParamsModel:
        if "params" not in self.model_fields_set:
            self._resolve_method_payload(self.__dict__["method"], None)
        return self


class InitialConfigRequestModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _validate_config_option_tags(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        options = value.get("configOptions", value.get("config_options"))
        if isinstance(options, list | tuple):
            for option in options:
                if isinstance(option, dict) and "type" not in option:
                    raise ValueError("configOptions entries require an explicit 'type' field")
        return value

    @model_validator(mode="after")
    def _validate_config_ids(self) -> InitialConfigRequestModel:
        options = self.__dict__.get("config_options", ())
        if len({option.config_id for option in options}) != len(options):
            raise ValueError("configOptions contains duplicate configId values")
        return self
