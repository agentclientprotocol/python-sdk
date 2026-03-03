"""Tests for the ErrorCode enum and its integration with RequestError."""

from __future__ import annotations

import pytest

from acp.exceptions import ErrorCode, RequestError


class TestErrorCodeValues:
    """Verify enum values match the ACP specification."""

    def test_parse_error(self) -> None:
        assert ErrorCode.PARSE_ERROR == -32700

    def test_invalid_request(self) -> None:
        assert ErrorCode.INVALID_REQUEST == -32600

    def test_method_not_found(self) -> None:
        assert ErrorCode.METHOD_NOT_FOUND == -32601

    def test_invalid_params(self) -> None:
        assert ErrorCode.INVALID_PARAMS == -32602

    def test_internal_error(self) -> None:
        assert ErrorCode.INTERNAL_ERROR == -32603

    def test_request_cancelled(self) -> None:
        assert ErrorCode.REQUEST_CANCELLED == -32800

    def test_auth_required(self) -> None:
        assert ErrorCode.AUTH_REQUIRED == -32000

    def test_resource_not_found(self) -> None:
        assert ErrorCode.RESOURCE_NOT_FOUND == -32002


class TestBackwardCompatibility:
    """ErrorCode values must compare equal to raw integers for backward compatibility."""

    @pytest.mark.parametrize(
        ("code", "raw"),
        [
            (ErrorCode.PARSE_ERROR, -32700),
            (ErrorCode.INVALID_REQUEST, -32600),
            (ErrorCode.METHOD_NOT_FOUND, -32601),
            (ErrorCode.INVALID_PARAMS, -32602),
            (ErrorCode.INTERNAL_ERROR, -32603),
            (ErrorCode.REQUEST_CANCELLED, -32800),
            (ErrorCode.AUTH_REQUIRED, -32000),
            (ErrorCode.RESOURCE_NOT_FOUND, -32002),
        ],
    )
    def test_int_equality(self, code: ErrorCode, raw: int) -> None:
        assert code == raw
        assert raw == code

    def test_usable_in_arithmetic(self) -> None:
        assert ErrorCode.PARSE_ERROR + 100 == -32600

    def test_hashable_with_int(self) -> None:
        d: dict[int, str] = {-32700: "parse"}
        assert d[ErrorCode.PARSE_ERROR] == "parse"


class TestRequestErrorIntegration:
    """Verify that RequestError factory methods use ErrorCode values."""

    def test_parse_error_code(self) -> None:
        err = RequestError.parse_error()
        assert err.code == ErrorCode.PARSE_ERROR

    def test_invalid_request_code(self) -> None:
        err = RequestError.invalid_request()
        assert err.code == ErrorCode.INVALID_REQUEST

    def test_method_not_found_code(self) -> None:
        err = RequestError.method_not_found("foo")
        assert err.code == ErrorCode.METHOD_NOT_FOUND

    def test_invalid_params_code(self) -> None:
        err = RequestError.invalid_params()
        assert err.code == ErrorCode.INVALID_PARAMS

    def test_internal_error_code(self) -> None:
        err = RequestError.internal_error()
        assert err.code == ErrorCode.INTERNAL_ERROR

    def test_auth_required_code(self) -> None:
        err = RequestError.auth_required()
        assert err.code == ErrorCode.AUTH_REQUIRED

    def test_resource_not_found_code(self) -> None:
        err = RequestError.resource_not_found("file:///missing")
        assert err.code == ErrorCode.RESOURCE_NOT_FOUND

    def test_to_error_obj_has_int_code(self) -> None:
        err = RequestError.parse_error({"detail": "bad json"})
        obj = err.to_error_obj()
        assert obj["code"] == -32700
        assert obj["message"] == "Parse error"
        assert obj["data"] == {"detail": "bad json"}


class TestPublicExport:
    """Verify ErrorCode is accessible from the top-level package."""

    def test_import_from_acp(self) -> None:
        from acp import ErrorCode as EC

        assert EC.PARSE_ERROR == -32700
