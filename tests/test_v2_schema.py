import pytest
from pydantic import TypeAdapter, ValidationError

from acp.experimental.v2 import PROTOCOL_VERSION, schema
from acp.experimental.v2.interfaces import CreateElicitationRequest
from acp.experimental.v2.meta import PROTOCOL_METHODS
from acp.experimental.v2.schema import (
    AgentMessageChunk,
    IdleSessionStateUpdate,
    MessageMcpRequest,
    OtherSessionUpdate,
    TextContentBlock,
    UpdateSessionNotification,
)


def test_v2_models_fill_protocol_discriminators() -> None:
    update = AgentMessageChunk(
        message_id="message-1",
        content=TextContentBlock(text="hello"),
    )

    assert PROTOCOL_VERSION == 2
    assert PROTOCOL_METHODS == {"cancel_request": "$/cancel_request"}
    assert update.model_dump(by_alias=True, exclude_none=True) == {
        "sessionUpdate": "agent_message_chunk",
        "messageId": "message-1",
        "content": {"type": "text", "text": "hello"},
    }


def test_v2_open_union_preserves_unknown_updates() -> None:
    notification = UpdateSessionNotification.model_validate({
        "sessionId": "session-1",
        "update": {"sessionUpdate": "_vendor_status", "status": "waiting"},
    })

    assert isinstance(notification.update, OtherSessionUpdate)
    assert notification.model_dump(by_alias=True, exclude_none=True)["update"] == {
        "sessionUpdate": "_vendor_status",
        "status": "waiting",
    }


def test_v2_open_union_rejects_malformed_known_updates() -> None:
    with pytest.raises(ValidationError):
        UpdateSessionNotification.model_validate({
            "sessionId": "session-1",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "hello"},
            },
        })


def test_idle_state_preserves_failure_details_and_background_snapshot() -> None:
    wire = {
        "sessionId": "session-1",
        "update": {
            "sessionUpdate": "state_update",
            "state": "idle",
            "stopReason": "error",
            "error": {"code": -32603, "message": "failed"},
            "background": [],
        },
    }
    notification = UpdateSessionNotification.model_validate(wire)
    assert isinstance(notification.update, IdleSessionStateUpdate)
    assert notification.model_dump(by_alias=True, exclude_unset=True) == wire
    with pytest.raises(ValidationError):
        UpdateSessionNotification.model_validate({
            "sessionId": "session-1",
            "update": {
                "sessionUpdate": "state_update",
                "state": "idle",
                "stopReason": "end_turn",
                "error": {"code": -32603, "message": "unexpected"},
            },
        })


@pytest.mark.parametrize("mode", ["form", "url", "vendor/custom"])
@pytest.mark.parametrize("request_id", [7, None])
def test_elicitation_rejects_overlapping_scopes(mode: str, request_id: int | None) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(CreateElicitationRequest).validate_python({
            "mode": mode,
            "message": "Input",
            "sessionId": "session-1",
            "requestId": request_id,
            "requestedSchema": {"type": "object", "properties": {}},
            "elicitationId": "e",
            "url": "https://example.com/",
        })


def test_mcp_tool_call_attribution_requires_a_session() -> None:
    with pytest.raises(ValidationError, match="tool_call_id requires session_id"):
        MessageMcpRequest.model_validate({
            "serverId": "s",
            "requestId": "r",
            "method": "tools/call",
            "toolCallId": "call",
        })


def test_request_envelopes_select_params_by_method_and_preserve_extensions() -> None:
    params = {"sessionId": "s", "cwd": "/work", "cursorId": "cursor"}
    request = schema.ClientRequest.model_validate({"id": 1, "method": "session/fork", "params": params})
    assert isinstance(request.params, schema.ForkSessionRequest)
    assert request.model_dump(by_alias=True, exclude_unset=True)["params"] == params
    with pytest.raises(ValidationError):
        schema.ClientRequest.model_validate({"id": 1, "method": "session/prompt", "params": params})
    with pytest.raises(ValidationError):
        schema.ClientRequest.model_validate({"id": 1, "method": "session/prompt"})
    with pytest.raises(ValidationError):
        schema.ClientRequest.model_validate({"id": 1, "method": "session/prompt", "params": None})
    with pytest.raises(ValidationError):
        schema.ClientRequest.model_validate({"id": 1, "method": "session/cancel", "params": {"sessionId": "s"}})
    with pytest.raises(ValidationError):
        schema.ClientRequest.model_validate({"id": 1, "method": "unknown/method", "params": {}})

    logout = schema.ClientRequest.model_validate({"id": 2, "method": "auth/logout"})
    assert logout.params is None

    extension_params = {**params, "opaque": True}
    extension = schema.ClientRequest.model_validate({"id": 1, "method": "_vendor/custom", "params": extension_params})
    assert extension.model_dump(by_alias=True, exclude_unset=True)["params"] == extension_params


def test_method_dispatch_distinguishes_request_and_notification_payloads() -> None:
    params = {"serverId": "mcp", "requestId": "r", "method": "tools/list", "sessionId": "s"}
    request = schema.AgentRequest.model_validate({"id": 1, "method": "mcp/message", "params": params})
    notification = schema.ClientNotification.model_validate({"method": "mcp/message", "params": params})
    assert isinstance(request.params, schema.MessageMcpRequest)
    assert request.params.session_id == "s"
    assert isinstance(notification.params, schema.MessageMcpNotification)

    proxy = schema.ProxyRequest.model_validate({
        "id": 1,
        "method": "proxy/successor",
        "params": {"method": "session/list", "params": {}},
    })
    assert isinstance(proxy.params, schema.ProxySuccessorRequest)
    with pytest.raises(ValidationError):
        schema.ProxyRequest.model_validate({"id": 1, "method": "proxy/initialize"})
    with pytest.raises(ValidationError):
        schema.AgentNotification.model_validate({"method": "session/update", "params": None})


def test_proxy_response_preserves_an_opaque_successor_result() -> None:
    wire = {
        "id": 1,
        "result": {
            "protocolVersion": 2,
            "info": {"name": "agent", "version": "1"},
            "forwardedField": True,
        },
    }
    response = schema.ProxyResponse.model_validate(wire)
    assert response.model_dump(by_alias=True, exclude_unset=True) == wire


def test_available_commands_skip_invalid_entries_and_preserve_extensions() -> None:
    commands = [
        {"type": "command", "name": "plan", "description": "Make a plan"},
        {"type": "command", "name": "missing-description"},
        {"type": "_vendor/command", "name": "custom", "description": "Custom command", "opaque": 1},
    ]
    update = UpdateSessionNotification.model_validate({
        "sessionId": "s",
        "update": {"sessionUpdate": "available_commands_update", "availableCommands": commands},
    })
    assert update.model_dump(by_alias=True, exclude_unset=True)["update"]["availableCommands"] == [
        commands[0],
        commands[2],
    ]


def test_cursor_anchor_resolves_known_types_and_preserves_unknown_ones() -> None:
    known = schema.CursorUpdate.model_validate({
        "cursorId": "c",
        "anchor": {"type": "user_message", "messageId": "m", "position": "before"},
    })
    assert isinstance(known.anchor, schema.UserMessageCursorAnchor)
    with pytest.raises(ValidationError):
        schema.CursorUpdate.model_validate({"cursorId": "c", "anchor": {"type": "user_message"}})

    custom = {"type": "_vendor/custom", "offset": 5}
    unknown = schema.CursorUpdate.model_validate({"cursorId": "c", "anchor": custom})
    assert isinstance(unknown.anchor, schema.OtherCursorAnchor)
    assert unknown.model_dump(by_alias=True, exclude_unset=True)["anchor"] == custom


def test_default_on_null_normalizes_only_annotated_lists() -> None:
    omitted = schema.NewSessionRequest(cwd="/work")
    explicit_null = schema.NewSessionRequest.model_validate({
        "cwd": "/work",
        "mcpServers": None,
        "additionalDirectories": None,
        "configOptions": None,
    })
    assert omitted == explicit_null
    assert explicit_null.mcp_servers == explicit_null.additional_directories == explicit_null.config_options == []
    assert explicit_null.model_dump(by_alias=True, exclude_unset=True)["mcpServers"] == []
    with pytest.raises(ValidationError):
        schema.NewSessionRequest.model_validate({"cwd": "/work", "mcpServers": {}})


def test_config_options_reject_duplicate_ids_and_malformed_boolean_values() -> None:
    option = {"type": "boolean", "configId": "thinking", "value": True}
    with pytest.raises(ValidationError, match="duplicate configId"):
        schema.NewSessionRequest.model_validate({"cwd": "/work", "configOptions": [option, option]})
    with pytest.raises(ValidationError):
        schema.NewSessionRequest.model_validate({"cwd": "/work", "configOptions": [{**option, "value": "true"}]})


def test_version_error_requires_typed_data_and_other_errors_preserve_opaque_data() -> None:
    with pytest.raises(ValidationError):
        schema.Error.model_validate({"code": -32022, "message": "Version unavailable"})
    data = {"requested": 1, "supported": [2]}
    version_error = schema.Error.model_validate({"code": -32022, "message": "Version unavailable", "data": data})
    assert isinstance(version_error.data, schema.UnsupportedProtocolVersionErrorData)
    assert version_error.model_dump(mode="json", by_alias=True, exclude_unset=True)["data"] == data
    opaque = {**data, "vendor": True}
    other_error = schema.Error.model_validate({"code": -32603, "message": "Failed", "data": opaque})
    assert other_error.data == opaque


def test_accepted_elicitation_content_preserves_custom_json_values() -> None:
    content = {"name": "acp", "tags": ["a", "b"], "custom": {"nested": [1, None, True]}}
    response = TypeAdapter(schema.CreateElicitationResponse).validate_python({"action": "accept", "content": content})
    assert isinstance(response, schema.AcceptElicitationResponse)
    assert response.content == content


def test_select_config_options_use_plain_non_empty_lists() -> None:
    option = schema.SessionConfigSelectOption(value="fast", name="Fast")
    select = schema.SessionConfigSelect(current_value="fast", options=[option])
    assert select.options == [option]
    assert len(select.options) == 1
    with pytest.raises(ValidationError):
        schema.SessionConfigSelect(current_value="fast", options=[])


def test_prompt_idempotency_key_accepts_bounded_strings() -> None:
    request = schema.PromptRequest(
        session_id="s",
        prompt=[TextContentBlock(text="hi")],
        idempotency_key="retry-key",
    )
    assert request.idempotency_key == "retry-key"
    for invalid in ("", "k" * 256):
        with pytest.raises(ValidationError):
            schema.PromptRequest(session_id="s", prompt=[TextContentBlock(text="hi")], idempotency_key=invalid)
