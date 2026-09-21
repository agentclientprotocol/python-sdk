from __future__ import annotations

from typing import Any, TypeVar

from pydantic import AnyUrl, BaseModel, TypeAdapter

from . import schema
from .interfaces import CreateElicitationRequest, SetConfigOptionRequest

ModelT = TypeVar("ModelT", bound=BaseModel)
_CONFIG = TypeAdapter(SetConfigOptionRequest)
_ELICITATION = TypeAdapter(CreateElicitationRequest)


def build_request(model: type[ModelT], fields: dict[str, Any], meta: dict[str, Any]) -> ModelT:
    """Build a request without changing nested models' explicitly set fields."""
    params = {
        name: value for name, value in fields.items() if value is not None or model.model_fields[name].is_required()
    }
    if meta:
        params["field_meta"] = meta
    return model.model_validate(params)


def build_config_request(
    config_id: str, session_id: str, value: Any, config_type: str | None, meta: dict[str, Any]
) -> SetConfigOptionRequest:
    return _CONFIG.validate_python({
        "configId": config_id,
        "sessionId": session_id,
        "value": value,
        "type": config_type if config_type is not None else ("boolean" if isinstance(value, bool) else "id"),
        "_meta": meta or None,
    })


def build_elicitation_request(
    *,
    message: str,
    mode: str,
    session_id: str | None,
    request_id: int | str | None,
    tool_call_id: str | None,
    requested_schema: schema.ElicitationSchema | None,
    elicitation_id: str | None,
    url: str | AnyUrl | None,
    meta: dict[str, Any],
) -> CreateElicitationRequest:
    if session_id is not None and request_id is not None:
        raise ValueError("Specify either session_id or request_id")
    if session_id is None and tool_call_id is not None:
        raise ValueError("tool_call_id requires session_id")
    params: dict[str, Any] = {"message": message, "mode": mode}
    if session_id is not None:
        params["sessionId"] = session_id
        if tool_call_id is not None:
            params["toolCallId"] = tool_call_id
    else:
        params["requestId"] = request_id
    for name, value in (("requestedSchema", requested_schema), ("elicitationId", elicitation_id), ("url", url)):
        if value is not None:
            params[name] = value
    if meta:
        params["_meta"] = meta
    return _ELICITATION.validate_python(params)
