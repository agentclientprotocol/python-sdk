from __future__ import annotations

from enum import IntEnum
from typing import Any

__all__ = ["ErrorCode", "RequestError"]


class ErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 and ACP error codes.

    These match the error codes defined in the ACP specification and are
    consistent with the Rust and TypeScript SDKs.
    """

    # JSON-RPC 2.0 standard errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # ACP extension errors
    REQUEST_CANCELLED = -32800
    AUTH_REQUIRED = -32000
    RESOURCE_NOT_FOUND = -32002


class RequestError(Exception):
    """JSON-RPC 2.0 error helper."""

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data

    @classmethod
    def parse_error(cls, data: dict[str, Any] | None = None) -> RequestError:
        return cls(ErrorCode.PARSE_ERROR, "Parse error", data)

    @classmethod
    def invalid_request(cls, data: dict[str, Any] | None = None) -> RequestError:
        return cls(ErrorCode.INVALID_REQUEST, "Invalid request", data)

    @classmethod
    def method_not_found(cls, method: str) -> RequestError:
        return cls(ErrorCode.METHOD_NOT_FOUND, "Method not found", {"method": method})

    @classmethod
    def invalid_params(cls, data: dict[str, Any] | None = None) -> RequestError:
        return cls(ErrorCode.INVALID_PARAMS, "Invalid params", data)

    @classmethod
    def internal_error(cls, data: dict[str, Any] | None = None) -> RequestError:
        return cls(ErrorCode.INTERNAL_ERROR, "Internal error", data)

    @classmethod
    def auth_required(cls, data: dict[str, Any] | None = None) -> RequestError:
        return cls(ErrorCode.AUTH_REQUIRED, "Authentication required", data)

    @classmethod
    def resource_not_found(cls, uri: str | None = None) -> RequestError:
        data = {"uri": uri} if uri is not None else None
        return cls(ErrorCode.RESOURCE_NOT_FOUND, "Resource not found", data)

    def to_error_obj(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "data": self.data}
