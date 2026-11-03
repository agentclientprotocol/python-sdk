from __future__ import annotations

import pytest

from acp.contrib.tool_calls import ToolCallTracker
from acp.schema import ContentToolCallContent, TextContentBlock, ToolCallProgress


def test_tool_call_tracker_generates_ids_and_updates():
    tracker = ToolCallTracker(id_factory=lambda: "generated-id")
    start = tracker.start("external", title="Run command", name="shell")
    assert start.tool_call_id == "generated-id"
    assert start.name == "shell"
    progress = tracker.progress("external", name="terminal", status="completed")
    assert isinstance(progress, ToolCallProgress)
    assert progress.tool_call_id == "generated-id"
    assert progress.name == "terminal"
    view = tracker.view("external")
    assert view.name == "terminal"
    assert view.status == "completed"


def test_tool_call_tracker_streaming_text_updates_content():
    tracker = ToolCallTracker(id_factory=lambda: "stream-id")
    tracker.start("external", title="Stream", status="in_progress")
    update1 = tracker.append_stream_text("external", "hello")
    assert update1.content is not None
    first_content = update1.content[0]
    assert isinstance(first_content, ContentToolCallContent)
    assert isinstance(first_content.content, TextContentBlock)
    assert first_content.content.text == "hello"
    update2 = tracker.append_stream_text("external", ", world", status="in_progress")
    assert update2.content is not None
    second_content = update2.content[0]
    assert isinstance(second_content, ContentToolCallContent)
    assert isinstance(second_content.content, TextContentBlock)
    assert second_content.content.text == "hello, world"


@pytest.mark.parametrize("field", ["raw_input", "raw_output"])
@pytest.mark.parametrize("operation", ["start", "progress"])
def test_tracker_snapshots_incoming_raw_payloads(field: str, operation: str) -> None:
    tracker = ToolCallTracker()
    payload = {"nested": ["original"]}
    if operation == "start":
        notification = tracker.start(
            "external",
            title="Run",
            raw_input=payload if field == "raw_input" else None,
            raw_output=payload if field == "raw_output" else None,
        )
    else:
        tracker.start("external", title="Run")
        notification = tracker.progress(
            "external",
            raw_input=payload if field == "raw_input" else None,
            raw_output=payload if field == "raw_output" else None,
        )

    payload["nested"].append("caller mutation")

    assert getattr(tracker.view("external"), field) == {"nested": ["original"]}
    assert getattr(notification, field) == {"nested": ["original"]}


@pytest.mark.parametrize("field", ["raw_input", "raw_output"])
@pytest.mark.parametrize("output", ["start", "progress", "view", "tool_call_model"])
def test_tracker_raw_payload_exports_cannot_mutate_state(field: str, output: str) -> None:
    tracker = ToolCallTracker()
    payload = {"nested": ["original"]}
    notification = tracker.start(
        "external",
        title="Run",
        raw_input=payload if field == "raw_input" else None,
        raw_output=payload if field == "raw_output" else None,
    )
    if output == "progress":
        exported = tracker.progress(
            "external",
            raw_input=payload if field == "raw_input" else None,
            raw_output=payload if field == "raw_output" else None,
        )
    elif output == "view":
        exported = tracker.view("external")
    elif output == "tool_call_model":
        exported = tracker.tool_call_model("external")
    else:
        exported = notification

    getattr(exported, field)["nested"].append("consumer mutation")

    assert getattr(tracker.view("external"), field) == {"nested": ["original"]}


@pytest.mark.parametrize("payload", [None, False, 0, "text", ["item"]])
def test_tracker_preserves_json_raw_payload_values(payload: object) -> None:
    tracker = ToolCallTracker()
    tracker.start("external", title="Run", raw_input=payload, raw_output=payload)

    assert tracker.view("external").raw_input == payload
    assert tracker.tool_call_model("external").raw_output == payload


def test_tracker_progress_can_clear_raw_payloads() -> None:
    tracker = ToolCallTracker()
    tracker.start("external", title="Run", raw_input={"old": True}, raw_output={"old": True})

    progress = tracker.progress("external", raw_input=None, raw_output=None)

    assert progress.raw_input is None
    assert progress.raw_output is None
    assert tracker.view("external").raw_input is None
    assert tracker.view("external").raw_output is None
