"""Server-Sent Events (SSE) serialization + parsing building blocks.

Minimal helpers shared by the Streamable HTTP client and server.  We only need
the ``data:`` field (JSON-RPC payloads) plus keepalive comments; ``event:``,
``id:``, and ``retry:`` are not used by this transport (resumability is v2).
"""

from __future__ import annotations

import codecs
import json
from typing import TYPE_CHECKING, Any

from ._jsonrpc import decode_message
from .exceptions import RequestError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

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


async def _decode_event(
    data_lines: list[str], on_error: Callable[[RequestError], Awaitable[None]] | None
) -> dict[str, Any] | None:
    """Decode buffered ``data:`` lines into a JSON object, or None to skip."""
    payload = "\n".join(data_lines)
    if not payload:
        return None
    try:
        return decode_message(payload)
    except RequestError as error:
        if on_error is not None:
            await on_error(error)
        return None


def _append_field(line: str, data_lines: list[str]) -> None:
    """Append a ``data:`` field's value to the buffer; ignore other fields."""
    field, _, value = line.partition(":")
    if value.startswith(" "):
        value = value[1:]
    if field == "data":
        data_lines.append(value)


async def parse_sse_stream(
    chunks: AsyncIterator[bytes], *, on_error: Callable[[RequestError], Awaitable[None]] | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Parse an SSE byte stream, yielding decoded JSON-RPC ``data:`` payloads.

    Comments (lines starting with ``:``) and non-``data`` fields are ignored.
    Invalid JSON and non-object JSON payloads are skipped.
    Multi-line ``data:`` fields are concatenated with newlines per the spec. A
    blank line dispatches the buffered event.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    data_lines: list[str] = []
    skip_lf = False

    async for chunk in chunks:
        text = decoder.decode(chunk)
        if not text:
            continue
        if skip_lf and text.startswith("\n"):
            text = text[1:]
        skip_lf = text.endswith("\r")
        buffer += text.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line == "":
                event = await _decode_event(data_lines, on_error)
                data_lines = []
                if event is not None:
                    yield event
            elif not line.startswith(":"):
                _append_field(line, data_lines)

    decoder.decode(b"", final=True)

    # Flush a trailing event with no terminating blank line.
    event = await _decode_event(data_lines, on_error)
    if event is not None:
        yield event
