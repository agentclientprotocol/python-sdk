import pytest
from pydantic import TypeAdapter, ValidationError

from acp.experimental.v2 import PROTOCOL_VERSION, schema
from acp.experimental.v2._schema_base import WIRE_VALIDATION_CONTEXT
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

    ignored_error = UpdateSessionNotification.model_validate({
        "sessionId": "session-1",
        "update": {
            "sessionUpdate": "state_update",
            "state": "idle",
            "stopReason": "end_turn",
            "error": {"code": -32603, "message": "unexpected"},
        },
    })
    assert isinstance(ignored_error.update, IdleSessionStateUpdate)
    assert ignored_error.update.stop_reason == "end_turn"
    assert ignored_error.update.error is None
    assert ignored_error.model_dump(by_alias=True, exclude_unset=True) == {
        "sessionId": "session-1",
        "update": {
            "sessionUpdate": "state_update",
            "state": "idle",
            "stopReason": "end_turn",
        },
    }


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
    for request_model, base in (
        (schema.NewSessionRequest, {"cwd": "/work"}),
        (schema.ForkSessionRequest, {"sessionId": "s", "cwd": "/work"}),
        (schema.ResumeSessionRequest, {"sessionId": "s", "cwd": "/work"}),
    ):
        with pytest.raises(ValidationError):
            request_model.model_validate(
                {**base, "configOptions": [{"configId": "x", "value": "x"}]},
                context=WIRE_VALIDATION_CONTEXT,
            )
    constructed = schema.NewSessionRequest(
        cwd="/work",
        config_options=[schema.InitialIdConfigOption(config_id="model", value="fast")],
    )
    assert constructed.config_options == [schema.InitialIdConfigOption(config_id="model", value="fast")]


def test_error_preserves_version_and_opaque_data() -> None:
    data = {"requested": 1, "supported": [2]}
    version_error = schema.Error.model_validate({"code": -32022, "message": "Version unavailable", "data": data})
    assert version_error.data == data
    parsed_data = schema.UnsupportedProtocolVersionErrorData.model_validate(version_error.data)
    assert parsed_data.model_dump(mode="json", by_alias=True, exclude_unset=True) == data
    no_data = schema.Error.model_validate({"code": -32022, "message": "Version unavailable"})
    assert no_data.data is None
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


def test_malformed_patch_fields_are_omitted_while_explicit_null_clears_stay() -> None:
    notification = UpdateSessionNotification.model_validate({
        "sessionId": "session-1",
        "update": {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tool-1",
            "name": 42,
            "title": None,
            "_meta": [],
        },
    })
    assert notification.model_dump(by_alias=True, exclude_unset=True) == {
        "sessionId": "session-1",
        "update": {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tool-1",
            "title": None,
        },
    }


def test_dual_deserialize_list_fields_fallback_and_filter() -> None:
    fallback = schema.NewSessionResponse.model_validate({"sessionId": "s", "availableCommands": 123})
    assert fallback.available_commands is None
    assert fallback.model_dump(by_alias=True, exclude_unset=True) == {"sessionId": "s"}

    filtered = schema.NewSessionResponse.model_validate({
        "sessionId": "s",
        "availableCommands": [
            {"type": "command", "name": "plan", "description": "Plan"},
            {"type": "command", "name": "missing-description"},
        ],
    })
    assert [cmd.name for cmd in filtered.available_commands or []] == ["plan"]


def test_reusable_v2_union_aliases_validate_known_variants() -> None:
    assert isinstance(
        TypeAdapter(schema.SessionUpdate).validate_python({
            "sessionUpdate": "state_update",
            "state": "idle",
            "stopReason": "end_turn",
        }),
        IdleSessionStateUpdate,
    )
    assert isinstance(
        TypeAdapter(schema.ContentBlock).validate_python({"type": "text", "text": "hi"}),
        TextContentBlock,
    )
    assert isinstance(
        TypeAdapter(schema.StateUpdate).validate_python({"state": "running"}),
        schema.RunningState,
    )
    assert isinstance(
        TypeAdapter(schema.BackgroundWork).validate_python({"type": "terminal", "terminalId": "t1"}),
        schema.TerminalBackgroundWork,
    )
    assert isinstance(
        TypeAdapter(schema.AvailableCommand).validate_python({
            "type": "command",
            "name": "plan",
            "description": "Make a plan",
        }),
        schema.SlashCommand,
    )


def test_wire_validation_requires_explicit_discriminator_tags() -> None:
    valid_ctor = schema.ResumeSessionRequest(session_id="s", cwd="/work", replay_from=schema.ReplayFromStartVariant())
    assert isinstance(valid_ctor.replay_from, schema.ReplayFromStartVariant)
    valid_wire = schema.ResumeSessionRequest.model_validate(
        {"sessionId": "s", "cwd": "/work", "replayFrom": {"type": "start"}},
        context=WIRE_VALIDATION_CONTEXT,
    )
    assert isinstance(valid_wire.replay_from, schema.ReplayFromStartVariant)
    valid_null = schema.ResumeSessionRequest.model_validate(
        {"sessionId": "s", "cwd": "/work", "replayFrom": None},
        context=WIRE_VALIDATION_CONTEXT,
    )
    assert valid_null.replay_from is None

    for malformed in ("start", {}, {"type": 5}):
        with pytest.raises(ValidationError):
            schema.ResumeSessionRequest.model_validate(
                {"sessionId": "s", "cwd": "/work", "replayFrom": malformed},
                context=WIRE_VALIDATION_CONTEXT,
            )

    with pytest.raises(ValidationError):
        schema.PromptRequest.model_validate(
            {"sessionId": "s", "prompt": [{"text": "hi"}]},
            context=WIRE_VALIDATION_CONTEXT,
        )
    with pytest.raises(ValidationError):
        UpdateSessionNotification.model_validate(
            {"sessionId": "s", "update": {"sessionUpdate": "state_update"}},
            context=WIRE_VALIDATION_CONTEXT,
        )
