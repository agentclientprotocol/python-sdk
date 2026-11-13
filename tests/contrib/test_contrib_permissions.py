from __future__ import annotations

import pytest

from acp.contrib.permissions import MissingPermissionOptionsError, PermissionBroker, default_permission_options
from acp.contrib.tool_calls import ToolCallTracker
from acp.schema import (
    AllowedOutcome,
    ContentToolCallContent,
    PermissionOption,
    RequestPermissionRequest,
    RequestPermissionResponse,
    TextContentBlock,
)


@pytest.mark.asyncio
async def test_permission_broker_uses_tracker_state():
    captured: dict[str, RequestPermissionRequest] = {}

    async def fake_requester(request: RequestPermissionRequest):
        captured["request"] = request
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=request.options[0].option_id, outcome="selected")
        )

    tracker = ToolCallTracker(id_factory=lambda: "perm-id")
    tracker.start("external", title="Need approval")
    broker = PermissionBroker("session", fake_requester, tracker=tracker)

    result = await broker.request_for("external", description="Perform sensitive action")
    assert isinstance(result.outcome, AllowedOutcome)
    assert result.outcome.option_id == captured["request"].options[0].option_id
    assert captured["request"].tool_call.content is not None
    last_content = captured["request"].tool_call.content[-1]
    assert isinstance(last_content, ContentToolCallContent)
    assert isinstance(last_content.content, TextContentBlock)
    assert last_content.content.text.startswith("Perform sensitive action")


@pytest.mark.asyncio
async def test_permission_broker_accepts_custom_options():
    tracker = ToolCallTracker(id_factory=lambda: "custom")
    tracker.start("external", title="Custom options")
    options = [
        PermissionOption(option_id="allow", name="Allow once", kind="allow_once"),
    ]
    recorded: list[str] = []

    async def requester(request: RequestPermissionRequest):
        recorded.append(request.options[0].option_id)
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=request.options[0].option_id, outcome="selected")
        )

    broker = PermissionBroker("session", requester, tracker=tracker)
    await broker.request_for("external", options=options)
    assert recorded == ["allow"]


@pytest.mark.asyncio
async def test_permission_broker_none_uses_standard_options():
    tracker = ToolCallTracker(id_factory=lambda: "standard")
    tracker.start("external", title="Standard options")
    recorded: list[list[str]] = []

    async def requester(request: RequestPermissionRequest):
        recorded.append([option.option_id for option in request.options])
        return RequestPermissionResponse(outcome=AllowedOutcome(option_id="approve", outcome="selected"))

    broker = PermissionBroker("session", requester, tracker=tracker, default_options=None)
    await broker.request_for("external", options=None)
    assert recorded == [["approve", "approve_for_session", "reject"]]


@pytest.mark.asyncio
async def test_permission_broker_custom_default_and_override():
    tracker = ToolCallTracker(id_factory=lambda: "reject-only")
    tracker.start("external", title="Reject only")
    reject = PermissionOption(option_id="reject", name="Reject", kind="reject_once")
    allow = PermissionOption(option_id="allow", name="Allow", kind="allow_once")
    recorded: list[list[str]] = []

    async def requester(request: RequestPermissionRequest):
        recorded.append([option.option_id for option in request.options])
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=request.options[0].option_id, outcome="selected")
        )

    broker = PermissionBroker("session", requester, tracker=tracker, default_options=[reject])
    await broker.request_for("external")
    await broker.request_for("external", options=[allow])
    assert recorded == [["reject"], ["allow"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(("default_options", "options"), [(None, []), ([], None)])
async def test_permission_broker_rejects_empty_option_sources(default_options, options):
    tracker = ToolCallTracker(id_factory=lambda: "empty")
    tracker.start("external", title="No options")
    recorded: list[RequestPermissionRequest] = []

    async def requester(request: RequestPermissionRequest):
        recorded.append(request)
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=request.options[0].option_id, outcome="selected")
        )

    broker = PermissionBroker("session", requester, tracker=tracker, default_options=default_options)
    with pytest.raises(MissingPermissionOptionsError, match="requires at least one permission option"):
        await broker.request_for("external", options=options)
    assert recorded == []


def test_default_permission_options_shape():
    options = default_permission_options()
    assert len(options) == 3
    assert {opt.option_id for opt in options} == {"approve", "approve_for_session", "reject"}
