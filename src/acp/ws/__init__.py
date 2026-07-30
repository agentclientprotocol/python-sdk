"""WebSocket transport for ACP (experimental).

Public exports are import-guarded behind the ``http`` extra (which provides the
``websockets`` dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["create_websocket_stream"]

if TYPE_CHECKING:
    from .client import create_websocket_stream


def __getattr__(name: str) -> Any:
    if name == "create_websocket_stream":
        from .client import create_websocket_stream

        return create_websocket_stream
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
