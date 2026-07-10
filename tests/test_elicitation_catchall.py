"""Regression tests for the schema-v1.18 "custom or future" catch-all union idiom.

These guard the codegen support in ``scripts/gen_schema.py``: unknown discriminator
values must resolve to the catch-all variant while preserving the raw payload, and
known values must still resolve to their specific typed variant.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from acp.schema import (
    AcceptElicitationResponse,
    CreateElicitationRequest,
    CreateElicitationResponse,
    CreateOtherElicitationRequest,
    ElicitationOtherPropertySchema,
    ElicitationStringPropertySchema,
    OtherElicitationResponse,
    OtherMultiSelectItems,
    StringMultiSelectItems,
    TitledMultiSelectItems,
)

_RESPONSE = TypeAdapter(CreateElicitationResponse)
_REQUEST = TypeAdapter(CreateElicitationRequest)


def test_known_elicitation_response_resolves_to_specific_variant() -> None:
    assert isinstance(_RESPONSE.validate_python({"action": "accept"}), AcceptElicitationResponse)


def test_custom_elicitation_response_falls_back_to_catchall() -> None:
    parsed = _RESPONSE.validate_python({"action": "x-snooze", "until": "later"})
    assert isinstance(parsed, OtherElicitationResponse)
    assert parsed.action == "x-snooze"
    assert parsed.model_dump(by_alias=True)["until"] == "later"


def test_malformed_known_variant_is_rejected_not_catchall() -> None:
    # A "form" request missing the required requestedSchema must fail validation rather
    # than silently degrade to the catch-all (restores the schema's dropped `not` clause).
    with pytest.raises(ValidationError):
        _REQUEST.validate_python({"mode": "form", "message": "hi", "sessionId": "s1"})


def test_custom_elicitation_request_preserves_mode_and_payload() -> None:
    parsed = _REQUEST.validate_python({
        "mode": "x-voice",
        "message": "speak now",
        "sessionId": "sess-1",
        "codec": "opus",
    })
    assert isinstance(parsed, CreateOtherElicitationRequest)
    assert parsed.mode == "x-voice"
    assert parsed.model_dump(by_alias=True)["codec"] == "opus"


def test_elicitation_property_schema_catchall() -> None:
    adapter = TypeAdapter(ElicitationStringPropertySchema | ElicitationOtherPropertySchema)
    assert isinstance(adapter.validate_python({"type": "string"}), ElicitationStringPropertySchema)
    custom = adapter.validate_python({"type": "x-slider", "min": 0, "max": 9})
    assert isinstance(custom, ElicitationOtherPropertySchema)
    assert custom.model_dump(by_alias=True)["max"] == 9


def test_multi_select_items_variants() -> None:
    adapter = TypeAdapter(StringMultiSelectItems | OtherMultiSelectItems | TitledMultiSelectItems)
    assert isinstance(adapter.validate_python({"type": "string", "enum": ["a", "b"]}), StringMultiSelectItems)
    assert isinstance(adapter.validate_python({"anyOf": [{"const": "a", "title": "A"}]}), TitledMultiSelectItems)
    assert isinstance(adapter.validate_python({"type": "x-chips", "note": "hi"}), OtherMultiSelectItems)
