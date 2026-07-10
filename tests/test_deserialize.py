"""Regression tests for lenient deserialization restored by ``acp._deserialize`` and the
validators ``scripts/gen_schema.py`` injects: ``x-deserialize-default-on-error`` (salvage a
malformed field to its default) and ``x-deserialize-skip-invalid-items`` (drop bad array
items). Mirrors the TypeScript SDK's ``src/schema-deserialize.test.ts``.
"""

from typing import Any

from pydantic import BaseModel, field_validator

from acp._deserialize import salvage_on_error, skip_invalid_items
from acp.schema import ReadTextFileRequest, ToolCallUpdate, WriteTextFileRequest


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
    salvaged = ReadTextFileRequest.model_validate({"sessionId": "s", "path": "/p", "line": "nan"})
    assert salvaged.line is None
    kept = ReadTextFileRequest.model_validate({"sessionId": "s", "path": "/p", "line": 5})
    assert kept.line == 5


def test_skip_invalid_items_on_generated_model() -> None:
    good = {"type": "content", "content": {"type": "text", "text": "ok"}}
    update = ToolCallUpdate.model_validate({"toolCallId": "t", "content": [good, {"bogus": 1}]})
    assert len(update.content or []) == 1
