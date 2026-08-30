from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel

from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    AudioContentBlock,
    BlobResourceContents,
    CancelNotification,
    ConfigOptionUpdate,
    ContentToolCallContent,
    DeniedOutcome,
    EmbeddedResourceContentBlock,
    FileEditToolCallContent,
    ImageContentBlock,
    InitializeRequest,
    InitializeResponse,
    NewSessionRequest,
    NewSessionResponse,
    PlanEntry,
    PromptRequest,
    ReadTextFileRequest,
    ReadTextFileResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
    ResourceContentBlock,
    SetSessionConfigOptionResponse,
    SetSessionConfigOptionSelectRequest,
    TerminalToolCallContent,
    TextContentBlock,
    TextResourceContents,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
    WriteTextFileRequest,
)

GOLDEN_DIR = Path(__file__).parent / "golden"

# Map each golden fixture to the concrete schema model it should conform to.
GOLDEN_CASES: dict[str, type[BaseModel]] = {
    "cancel_notification": CancelNotification,
    "content_audio": AudioContentBlock,
    "content_image": ImageContentBlock,
    "content_resource_blob": EmbeddedResourceContentBlock,
    "content_resource_link": ResourceContentBlock,
    "content_resource_text": EmbeddedResourceContentBlock,
    "content_text": TextContentBlock,
    "fs_read_text_file_request": ReadTextFileRequest,
    "fs_read_text_file_response": ReadTextFileResponse,
    "fs_write_text_file_request": WriteTextFileRequest,
    "initialize_request": InitializeRequest,
    "initialize_response": InitializeResponse,
    "new_session_request": NewSessionRequest,
    "new_session_response": NewSessionResponse,
    "permission_outcome_cancelled": DeniedOutcome,
    "permission_outcome_selected": AllowedOutcome,
    "prompt_request": PromptRequest,
    "request_permission_request": RequestPermissionRequest,
    "request_permission_response_selected": RequestPermissionResponse,
    "session_update_agent_message_chunk": AgentMessageChunk,
    "session_update_agent_thought_chunk": AgentThoughtChunk,
    "session_update_config_option_update": ConfigOptionUpdate,
    "session_update_plan": AgentPlanUpdate,
    "session_update_tool_call": ToolCallStart,
    "session_update_tool_call_edit": ToolCallStart,
    "session_update_tool_call_locations_rawinput": ToolCallStart,
    "session_update_tool_call_read": ToolCallStart,
    "session_update_tool_call_update_content": ToolCallProgress,
    "session_update_tool_call_update_more_fields": ToolCallProgress,
    "session_update_user_message_chunk": UserMessageChunk,
    "set_session_config_option_request": SetSessionConfigOptionSelectRequest,
    "set_session_config_option_response": SetSessionConfigOptionResponse,
    "tool_content_content_text": ContentToolCallContent,
    "tool_content_diff": FileEditToolCallContent,
    "tool_content_diff_no_old": FileEditToolCallContent,
    "tool_content_terminal": TerminalToolCallContent,
}

_PARAMS = tuple(sorted(GOLDEN_CASES.items()))
_PARAM_IDS = [name for name, _ in _PARAMS]

MODEL_BUILDERS: dict[str, Callable[[], BaseModel]] = {
    "content_text": lambda: TextContentBlock(text="What's the weather like today?"),
    "content_image": lambda: ImageContentBlock(
        data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB...",
        mime_type="image/png",
    ),
    "content_audio": lambda: AudioContentBlock(
        data="UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAAB...",
        mime_type="audio/wav",
    ),
    "content_resource_text": lambda: EmbeddedResourceContentBlock(
        resource=TextResourceContents(
            uri="file:///home/user/script.py",
            text="def hello():\n    print('Hello, world!')",
            mime_type="text/x-python",
        )
    ),
    "content_resource_blob": lambda: EmbeddedResourceContentBlock(
        resource=BlobResourceContents(
            uri="file:///home/user/document.pdf",
            blob="<b64>",
            mime_type="application/pdf",
        )
    ),
    "content_resource_link": lambda: ResourceContentBlock(
        name="document.pdf",
        uri="file:///home/user/document.pdf",
        mime_type="application/pdf",
        size=1_024_000,
    ),
    "tool_content_content_text": lambda: ContentToolCallContent(
        content=TextContentBlock(text="Analysis complete. Found 3 issues.")
    ),
    "tool_content_diff": lambda: FileEditToolCallContent(
        path="/home/user/project/src/config.json",
        new_text='{\n  "debug": true\n}',
        old_text='{\n  "debug": false\n}',
    ),
    "tool_content_diff_no_old": lambda: FileEditToolCallContent(
        path="/home/user/project/src/config.json",
        new_text='{\n  "debug": true\n}',
    ),
    "tool_content_terminal": lambda: TerminalToolCallContent(terminal_id="term_001"),
    "session_update_user_message_chunk": lambda: UserMessageChunk(
        content=TextContentBlock(text="What's the capital of France?")
    ),
    "session_update_agent_message_chunk": lambda: AgentMessageChunk(
        content=TextContentBlock(text="The capital of France is Paris.")
    ),
    "session_update_agent_thought_chunk": lambda: AgentThoughtChunk(
        content=TextContentBlock(text="Thinking about best approach...")
    ),
    "session_update_plan": lambda: AgentPlanUpdate(
        entries=[
            PlanEntry(
                content="Check for syntax errors",
                priority="high",
                status="pending",
            ),
            PlanEntry(
                content="Identify potential type issues",
                priority="medium",
                status="pending",
            ),
        ]
    ),
    "session_update_tool_call": lambda: ToolCallStart(
        tool_call_id="call_001",
        title="Reading configuration file",
        kind="read",
        status="pending",
    ),
    "session_update_tool_call_read": lambda: ToolCallStart(
        tool_call_id="call_001",
        title="Reading configuration file",
        kind="read",
        status="pending",
        locations=[ToolCallLocation(path="/home/user/project/src/config.json")],
        raw_input={"path": "/home/user/project/src/config.json"},
    ),
    "session_update_tool_call_edit": lambda: ToolCallStart(
        tool_call_id="call_003",
        title="Apply edit",
        kind="edit",
        status="pending",
        locations=[ToolCallLocation(path="/home/user/project/src/config.json")],
        raw_input={
            "path": "/home/user/project/src/config.json",
            "content": "print('hello')",
        },
    ),
    "session_update_tool_call_locations_rawinput": lambda: ToolCallStart(
        tool_call_id="call_lr",
        title="Tracking file",
        locations=[ToolCallLocation(path="/home/user/project/src/config.json")],
        raw_input={"path": "/home/user/project/src/config.json"},
    ),
    "session_update_tool_call_update_content": lambda: ToolCallProgress(
        tool_call_id="call_001",
        status="in_progress",
        content=[ContentToolCallContent(content=TextContentBlock(text="Found 3 configuration files..."))],
    ),
    "session_update_tool_call_update_more_fields": lambda: ToolCallProgress(
        tool_call_id="call_010",
        title="Processing changes",
        kind="edit",
        status="completed",
        locations=[ToolCallLocation(path="/home/user/project/src/config.json")],
        raw_input={"path": "/home/user/project/src/config.json"},
        raw_output={"result": "ok"},
        content=[ContentToolCallContent(content=TextContentBlock(text="Edit completed."))],
    ),
}

_MODEL_PARAMS = tuple(sorted(MODEL_BUILDERS.items()))
_MODEL_IDS = [name for name, _ in _MODEL_PARAMS]


def _load_golden(name: str) -> dict:
    path = GOLDEN_DIR / f"{name}.json"
    return json.loads(path.read_text())


def _dump_model(model: BaseModel) -> dict:
    return model.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True)


def test_golden_cases_covered() -> None:
    available = {path.stem for path in GOLDEN_DIR.glob("*.json")}
    assert available == set(GOLDEN_CASES), "Add the new golden file to GOLDEN_CASES."


@pytest.mark.parametrize(
    ("name", "model_cls"),
    _PARAMS,
    ids=_PARAM_IDS,
)
def test_json_golden_roundtrip(name: str, model_cls: type[BaseModel]) -> None:
    raw = _load_golden(name)
    model = model_cls.model_validate(raw)
    assert _dump_model(model) == raw


@pytest.mark.parametrize(
    ("name", "builder"),
    _MODEL_PARAMS,
    ids=_MODEL_IDS,
)
def test_models_match_golden(name: str, builder: Callable[[], BaseModel]) -> None:
    raw = _load_golden(name)
    model = builder()
    assert isinstance(model, BaseModel)
    assert _dump_model(model) == raw
