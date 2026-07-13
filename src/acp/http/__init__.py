"""Streamable HTTP transport for ACP (experimental).

Public exports are import-guarded: the heavy client/server implementations pull
in optional dependencies (``httpx[http2]``).  Importing a symbol without the
extra installed raises a friendly ``ImportError`` pointing at
``pip install agent-client-protocol[http]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["AcpServer", "create_http_stream"]

if TYPE_CHECKING:
    from .client import create_http_stream
    from .server import AcpServer


def __getattr__(name: str) -> Any:
    if name == "create_http_stream":
        from .client import create_http_stream

        return create_http_stream
    if name == "AcpServer":
        from .server import AcpServer

        return AcpServer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
