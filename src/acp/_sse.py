"""Server-Sent Events (SSE) serialization + parsing building blocks.

Minimal helpers shared by the Streamable HTTP client and server.  We only need
the ``data:`` field (JSON-RPC payloads) plus keepalive comments; ``event:``,
``id:``, and ``retry:`` are not used by this transport (resumability is v2).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = [
    "parse_sse_stream",
    "serialize_sse_event",
    "serialize_sse_keepalive",
]


def serialize_sse_event(message: dict[str, Any]) -> bytes:
    """Serialize a JSON-RPC message as an SSE ``data:`` event.

    The payload is JSON-encoded on a single line and terminated by a blank line,
    per the SSE framing rules.
    """
    data = json.dumps(message, separators=(",", ":"))
    return f"data: {data}\n\n".encode()


def serialize_sse_keepalive() -> bytes:
    """Serialize an SSE comment used to keep the connection alive."""
    return b": keepalive\n\n"


def _decode_event(data_lines: list[str]) -> dict[str, Any] | None:
    """Decode buffered ``data:`` lines into a JSON object, or None to skip."""
    payload = "\n".join(data_lines)
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _append_field(line: str, data_lines: list[str]) -> None:
    """Append a ``data:`` field's value to the buffer; ignore other fields."""
    field, _, value = line.partition(":")
    if value.startswith(" "):
        value = value[1:]
    if field == "data":
        data_lines.append(value)


async def parse_sse_stream(chunks: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    """Parse an SSE byte stream, yielding decoded JSON-RPC ``data:`` payloads.

    Comments (lines starting with ``:``) and non-``data`` fields are ignored.
    Multi-line ``data:`` fields are concatenated with newlines per the spec. A
    blank line dispatches the buffered event.
    """
    buffer = ""
    data_lines: list[str] = []

    async for chunk in chunks:
        buffer += chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line == "":
                event = _decode_event(data_lines)
                data_lines = []
                if event is not None:
                    yield event
            elif not line.startswith(":"):
                _append_field(line, data_lines)

    # Flush a trailing event with no terminating blank line.
    event = _decode_event(data_lines)
    if event is not None:
        yield event
