from __future__ import annotations

import json
from typing import Any

from .exceptions import RequestError


def decode_message(payload: str | bytes) -> dict[str, Any] | None:
    """Decode a message, returning None for unsupported non-empty batches."""
    try:
        message = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RequestError.parse_error() from exc
    return require_message_object(message)


def require_message_object(message: Any) -> dict[str, Any] | None:
    """Reject invalid root values while leaving non-empty batches unsupported."""
    if isinstance(message, dict) and message:
        return message
    if isinstance(message, list) and message:
        return None
    raise RequestError.invalid_request()


def error_response(error: RequestError) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": None, "error": error.to_error_obj()}
