from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from acp._sse import parse_sse_stream, serialize_sse_event, serialize_sse_keepalive


def test_serialize_sse_event_frames_json_with_blank_line() -> None:
    frame = serialize_sse_event({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
    assert frame == b'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'


def test_serialize_sse_keepalive_is_a_comment() -> None:
    assert serialize_sse_keepalive() == b": keepalive\n\n"


async def _aiter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_parse_single_event() -> None:
    stream = _aiter([b'data: {"id":1}\n\n'])
    events = [event async for event in parse_sse_stream(stream)]
    assert events == [{"id": 1}]


@pytest.mark.asyncio
async def test_parse_multiple_events_split_across_chunks() -> None:
    stream = _aiter([b'data: {"id', b'":1}\n\ndata: {"id":2}', b"\n\n"])
    events = [event async for event in parse_sse_stream(stream)]
    assert events == [{"id": 1}, {"id": 2}]


@pytest.mark.asyncio
async def test_parse_multibyte_utf8_split_across_chunks() -> None:
    frame = 'data: {"text":"你好"}\n\n'.encode()
    split_at = frame.index("你".encode()) + 1
    stream = _aiter([frame[:split_at], frame[split_at:]])

    events = [event async for event in parse_sse_stream(stream)]

    assert events == [{"text": "你好"}]


@pytest.mark.asyncio
async def test_parse_ignores_comments_and_other_fields() -> None:
    stream = _aiter([b': keepalive\n\nevent: message\ndata: {"id":7}\n\n'])
    events = [event async for event in parse_sse_stream(stream)]
    assert events == [{"id": 7}]


@pytest.mark.asyncio
async def test_parse_multiline_data_is_joined() -> None:
    stream = _aiter([b'data: {"a":1,\ndata: "b":2}\n\n'])
    events = [event async for event in parse_sse_stream(stream)]
    assert events == [{"a": 1, "b": 2}]


@pytest.mark.asyncio
async def test_parse_flushes_trailing_event_without_blank_line() -> None:
    stream = _aiter([b'data: {"id":9}\n'])
    events = [event async for event in parse_sse_stream(stream)]
    assert events == [{"id": 9}]


@pytest.mark.asyncio
async def test_parse_skips_invalid_json() -> None:
    stream = _aiter([b'data: not-json\n\ndata: {"id":1}\n\n'])
    events = [event async for event in parse_sse_stream(stream)]
    assert events == [{"id": 1}]
