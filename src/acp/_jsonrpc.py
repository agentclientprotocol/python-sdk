from __future__ import annotations

import json
from typing import Any

from .exceptions import RequestError

JsonRpcMessage = dict[str, Any] | list[Any]


def decode_message(payload: str | bytes) -> JsonRpcMessage | None:
    """Decode a JSON-RPC message, returning a non-empty batch as a list."""
    try:
        message = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RequestError.parse_error() from exc
    if isinstance(message, list):
        # An empty batch is not a batch: the spec answers it with a single -32600 object.
        if not message:
            raise RequestError.invalid_request()
        return message
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
