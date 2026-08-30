from __future__ import annotations

from typing import Any, ClassVar, Literal, get_args, get_origin

import pydantic
from pydantic import (
    ConfigDict,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.alias_generators import to_snake

from ._deserialize import use_default_on_error


class BaseModel(pydantic.BaseModel):
    """Runtime behavior shared by generated ACP schema models."""

    _reserved_tags: ClassVar[dict[str, tuple[str, frozenset[str]]]] = {
        "CreateElicitationRequest3": ("mode", frozenset({"form", "url"})),
        "CreateElicitationRequest4": ("mode", frozenset({"form", "url"})),
        "CreateElicitationResponse4": ("action", frozenset({"accept", "cancel", "decline"})),
        "ElicitationPropertySchema6": (
            "type",
            frozenset({"array", "boolean", "integer", "number", "string"}),
        ),
        "MultiSelectItems2": ("type", frozenset({"string"})),
    }

    model_config = ConfigDict(
        populate_by_name=True,
        use_attribute_docstrings=True,
    )

    def __getattr__(self, item: str) -> Any:
        if item.lower() != item:
            return getattr(self, to_snake(item))
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {item!r}")

    @model_serializer(mode="wrap")
    def _include_literal_defaults(
        self,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> Any:
        data = handler(self)
        for name, field in type(self).model_fields.items():
            annotation = field.annotation
            if info.include is not None and name not in info.include:
                continue
            if info.exclude is not None and name in info.exclude:
                continue
            if get_origin(annotation) is not Literal or len(get_args(annotation)) != 1:
                continue
            key = (field.serialization_alias or field.alias or name) if info.by_alias else name
            data[key] = getattr(self, name)
        return data

    @field_validator("field_meta", mode="wrap", check_fields=False)
    @classmethod
    def _use_meta_default_on_error(cls, value: Any, handler: Any) -> Any:
        return use_default_on_error(value, handler)

    @model_validator(mode="before")
    @classmethod
    def _reject_malformed_known_variant(cls, value: Any) -> Any:
        rule = cls._reserved_tags.get(cls.__name__)
        if rule is None or not isinstance(value, dict):
            return value
        wire_field, reserved = rule
        tag = value.get(wire_field, value.get(to_snake(wire_field)))
        if tag in reserved:
            raise ValueError(f"{wire_field} value is reserved by a known variant")
        return value
