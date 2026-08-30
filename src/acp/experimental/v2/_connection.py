from __future__ import annotations

import asyncio
from typing import Any

from acp._transport import Transport
from acp.connection import Connection, MethodHandler


def open_connection(
    handler: MethodHandler,
    input_stream: Any,
    output_stream: Any = None,
    *,
    listening: bool = True,
    **connection_kwargs: Any,
) -> Connection:
    if isinstance(input_stream, Transport):
        if output_stream is not None:
            raise TypeError("A message transport cannot be combined with an output stream")
        return Connection(handler, input_stream, listening=listening, **connection_kwargs)
    if not isinstance(input_stream, asyncio.StreamWriter) or not isinstance(output_stream, asyncio.StreamReader):
        raise TypeError("Expected an asyncio StreamWriter/StreamReader pair or a message transport")
    return Connection(handler, input_stream, output_stream, listening=listening, **connection_kwargs)
