from __future__ import annotations

from acp.http.protocol import (
    INITIALIZE_METHOD,
    is_initialize_request,
    is_response_message,
    message_id_key,
    method_requires_session_header,
    session_id_from_message,
    session_id_from_params,
    session_id_from_result,
)


def test_is_initialize_request() -> None:
    assert is_initialize_request({"method": INITIALIZE_METHOD, "id": 1, "params": {}})
    assert not is_initialize_request({"method": "session/new", "id": 2})
    # A notification (no id) is not a request.
    assert not is_initialize_request({"method": INITIALIZE_METHOD})


def test_is_response_message() -> None:
    assert is_response_message({"id": 1, "result": {}})
    assert is_response_message({"id": 1, "error": {"code": -1, "message": "x"}})
    assert not is_response_message({"id": 1, "method": "session/prompt"})
    assert not is_response_message({"method": "session/update", "params": {}})


def test_method_requires_session_header() -> None:
    assert method_requires_session_header("session/prompt")
    assert method_requires_session_header("session/cancel")
    assert method_requires_session_header("session/set_mode")
    # Connection-level methods (and session-establishing ones) do not require the header.
    assert not method_requires_session_header("initialize")
    assert not method_requires_session_header("session/new")
    assert not method_requires_session_header("session/load")
    assert not method_requires_session_header("session/list")
    assert not method_requires_session_header(None)


def test_message_id_key_normalizes_int_and_str() -> None:
    assert message_id_key(1) == "1"
    assert message_id_key("1") == "1"
    assert message_id_key(1) == message_id_key("1")
    assert message_id_key(None) is None


def test_session_id_extraction() -> None:
    assert session_id_from_params({"sessionId": "s1"}) == "s1"
    assert session_id_from_params({}) is None
    assert session_id_from_params(None) is None
    assert session_id_from_result({"sessionId": "s2"}) == "s2"
    assert session_id_from_message({"params": {"sessionId": "s3"}}) == "s3"
    assert session_id_from_message({"result": {"sessionId": "s4"}}) == "s4"
    assert session_id_from_message({"result": {}}) is None
