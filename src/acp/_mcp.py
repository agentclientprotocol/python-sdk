"""Validation of outgoing MCP outcomes, separate from ACP errors."""

from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from .exceptions import RequestError


def validate_mcp_response(payload: Any, *, adapter: TypeAdapter[Any]) -> Any:
    """Require exactly one outcome when sending; receivers may prefer result."""
    fields = type(payload).model_fields if isinstance(payload, BaseModel) else payload
    if not isinstance(fields, dict) or (("result" in fields) == ("error" in fields)):
        raise RequestError(-33002, "Invalid MCP response carrier: expected exactly one result or error")
    try:
        return adapter.validate_python(payload)
    except ValidationError as exc:
        raise RequestError(-33002, "Invalid MCP response carrier") from exc
