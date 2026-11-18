"""Regression tests for lenient deserialization restored by ``acp._deserialize`` and the
validators ``scripts/gen_schema.py`` injects: ``x-deserialize-default-on-error`` (salvage a
malformed field to its default) and ``x-deserialize-skip-invalid-items`` (drop bad array
items). Mirrors the TypeScript SDK's ``src/schema-deserialize.test.ts``.
"""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from acp._deserialize import salvage_on_error, skip_invalid_items
from acp.schema import (
    HttpHeader,
    HttpMcpServer,
    McpServerHttp,
    NewSessionRequest,
    ToolCallLocation,
    ToolCallUpdate,
    WriteTextFileRequest,
)


class _Salvage(BaseModel):
    n: int | None = None

    @field_validator("n", mode="wrap")
    @classmethod
    def _v(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


def test_salvage_on_error_replaces_invalid_value() -> None:
    assert _Salvage.model_validate({"n": "not-an-int"}).n is None
    assert _Salvage.model_validate({"n": 7}).n == 7


class _Skip(BaseModel):
    xs: list[int] = []

    @field_validator("xs", mode="wrap")
    @classmethod
    def _v(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


def test_skip_invalid_items_drops_bad_entries() -> None:
    assert _Skip.model_validate({"xs": [1, "bad", 3]}).xs == [1, 3]


def test_meta_is_salvaged_to_none() -> None:
    request = WriteTextFileRequest.model_validate({
        "sessionId": "s",
        "path": "/p",
        "content": "x",
        "_meta": "not-a-dict",
    })
    assert request.field_meta is None


def test_default_on_error_field_is_salvaged() -> None:
    # Schema 1.33 dropped x-deserialize-default-on-error from ReadTextFileRequest.line; ToolCallLocation.line keeps it.
    salvaged = ToolCallLocation.model_validate({"path": "/p", "line": "nan"})
    assert salvaged.line is None
    kept = ToolCallLocation.model_validate({"path": "/p", "line": 5})
    assert kept.line == 5


def test_skip_invalid_items_on_generated_model() -> None:
    good = {"type": "content", "content": {"type": "text", "text": "ok"}}
    update = ToolCallUpdate.model_validate({"toolCallId": "t", "content": [good, {"bogus": 1}]})
    assert len(update.content or []) == 1


class _Item(BaseModel):
    n: int


class _Other(BaseModel):
    s: str


class _SkipModels(BaseModel):
    xs: list[_Item] = []

    @field_validator("xs", mode="wrap")
    @classmethod
    def _v(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


def test_skip_invalid_items_raises_for_local_model_instances() -> None:
    # Wire data (dicts) stays lenient...
    assert _SkipModels.model_validate({"xs": [{"n": 1}, {"bogus": 1}]}).xs == [_Item(n=1)]
    # ...but a wrong model built in code is a programming error, not data from a newer peer.
    with pytest.raises(ValidationError):
        _SkipModels(xs=[_Item(n=1), _Other(s="x")])  # type: ignore[list-item]


def test_new_session_request_rejects_untagged_mcp_server_instance() -> None:
    headers = [HttpHeader(name="Authorization", value="Bearer x")]
    untagged = McpServerHttp(name="m", url="http://127.0.0.1:1/mcp", headers=headers)
    with pytest.raises(ValidationError):
        NewSessionRequest(cwd="/w", mcp_servers=[untagged])  # type: ignore[list-item]
    ok = NewSessionRequest(
        cwd="/w", mcp_servers=[HttpMcpServer(name="m", url="http://127.0.0.1:1/mcp", headers=headers)]
    )
    assert [s.name for s in ok.mcp_servers] == ["m"]
    # Incoming JSON is unchanged: the same fields as a dict parse (type defaults to "http"),
    # and since schema 1.33 (`mcpServers` is no longer `x-deserialize-skip-invalid-items`) a
    # malformed server is an error, not skipped.
    wire = NewSessionRequest.model_validate({"cwd": "/w", "mcpServers": [{"name": "m", "url": "u", "headers": []}]})
    assert [s.name for s in wire.mcp_servers] == ["m"]
    with pytest.raises(ValidationError):
        NewSessionRequest.model_validate({
            "cwd": "/w",
            "mcpServers": [{"name": "m", "url": "u", "headers": []}, {"bogus": 1}],
        })
