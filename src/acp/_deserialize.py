"""Runtime helpers that restore the lenient-deserialization semantics the ACP schema
declares via ``x-deserialize-default-on-error`` and ``x-deserialize-skip-invalid-items``
but that generated Pydantic models cannot express directly.

Referenced by ``field_validator`` methods that ``scripts/gen_schema.py`` injects into
``schema.py``. Mirrors the TypeScript SDK's ``src/schema-deserialize.ts``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import ValidationError


def salvage_on_error(value: Any, handler: Callable[[Any], Any], fallback: Callable[[], Any]) -> Any:
    """Return ``fallback()`` when ``value`` fails validation, otherwise the validated value.

    Restores ``x-deserialize-default-on-error``: a malformed non-critical field is replaced
    with its default rather than failing the whole payload.
    """
    try:
        return handler(value)
    except ValidationError:
        return fallback()


def skip_invalid_items(value: Any, handler: Callable[[Any], Any]) -> Any:
    """Drop array items that fail validation instead of failing the whole array.

    Restores ``x-deserialize-skip-invalid-items``. Each item is validated through the field's
    own list handler, so item coercion and salvaging still apply to the survivors.
    """
    if not isinstance(value, list):
        return handler(value)
    salvaged: list[Any] = []
    for item in value:
        try:
            salvaged.append(handler([item])[0])
        except ValidationError:
            continue
    return salvaged
