from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, TypeAdapter

from .exceptions import RequestError
from .schema import (
    CreateElicitationRequest,
    CreateFormRequestElicitationRequest,
    CreateFormSessionElicitationRequest,
    CreateUrlRequestElicitationRequest,
    CreateUrlSessionElicitationRequest,
    ElicitationFormRequestMode,
    ElicitationFormSessionMode,
    ElicitationUrlRequestMode,
    ElicitationUrlSessionMode,
    SetSessionConfigOptionBooleanRequest,
    SetSessionConfigOptionSelectRequest,
)

_CREATE_ELICITATION_REQUEST_ADAPTER = TypeAdapter(CreateElicitationRequest)


def validate_create_elicitation_request(params: Any) -> CreateElicitationRequest:
    return _CREATE_ELICITATION_REQUEST_ADAPTER.validate_python(params)


def _mode_from_create_elicitation_request(
    request: CreateElicitationRequest,
) -> ElicitationFormSessionMode | ElicitationFormRequestMode | ElicitationUrlSessionMode | ElicitationUrlRequestMode:
    if isinstance(request, CreateFormSessionElicitationRequest):
        return ElicitationFormSessionMode(
            session_id=request.session_id,
            tool_call_id=request.tool_call_id,
            requested_schema=request.requested_schema,
        )
    if isinstance(request, CreateFormRequestElicitationRequest):
        return ElicitationFormRequestMode(
            request_id=request.request_id,
            requested_schema=request.requested_schema,
        )

    if isinstance(request, CreateUrlSessionElicitationRequest):
        return ElicitationUrlSessionMode(
            session_id=request.session_id,
            tool_call_id=request.tool_call_id,
            elicitation_id=request.elicitation_id,
            url=request.url,
        )
    if isinstance(request, CreateUrlRequestElicitationRequest):
        return ElicitationUrlRequestMode(
            request_id=request.request_id,
            elicitation_id=request.elicitation_id,
            url=request.url,
        )
    raise RequestError.invalid_params({"details": f"Unsupported elicitation mode: {request.mode!r}"})


def elicitation_to_kwargs(request: BaseModel) -> dict[str, Any]:
    # The validator has already resolved the wire union, including custom modes.
    request = cast(CreateElicitationRequest, request)
    kwargs = {"message": request.message, "mode": _mode_from_create_elicitation_request(request)}
    if request.field_meta:
        kwargs.update(request.field_meta)
    return kwargs


def validate_set_config_option_request(params: Any) -> BaseModel:
    if isinstance(params, dict) and params.get("type") == "boolean":
        return SetSessionConfigOptionBooleanRequest.model_validate(params)
    return SetSessionConfigOptionSelectRequest.model_validate(params)
