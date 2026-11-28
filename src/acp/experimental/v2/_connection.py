from __future__ import annotations

import asyncio
from typing import Any

from acp._transport import NdjsonTransport, Transport
from acp.connection import Connection, MethodHandler
from acp.task import MessageSender


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
    transport = NdjsonTransport(
        output_stream,
        MessageSender(input_stream),
        receive_timeout=connection_kwargs.pop("receive_timeout", None),
        reject_batches=connection_kwargs.pop("reject_batches", True),
    )
    return Connection(handler, transport, listening=listening, **connection_kwargs)
