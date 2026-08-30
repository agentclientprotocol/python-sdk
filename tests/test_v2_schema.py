import pytest
from pydantic import ValidationError

from acp.experimental.v2 import PROTOCOL_VERSION
from acp.experimental.v2.schema import (
    AgentMessageChunk,
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
