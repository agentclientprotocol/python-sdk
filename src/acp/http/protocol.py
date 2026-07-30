"""Protocol constants and JSON-RPC routing helpers for the HTTP/WS transport.

Ports the small helpers from #155's ``protocol.ts`` + ``jsonrpc.ts``: header
names, MIME types, endpoint path, and pure functions that classify JSON-RPC
messages and extract routing keys (``id`` normalization, ``sessionId``).
"""

from __future__ import annotations

from typing import Any

from ..meta import AGENT_METHODS

__all__ = [
    "ACP_ENDPOINT_PATH",
    "CONNECTION_ID_HEADER",
    "CONTENT_TYPE_JSON",
    "CONTENT_TYPE_SSE",
    "INITIALIZE_METHOD",
    "SESSION_ID_HEADER",
    "is_initialize_request",
    "is_response_message",
    "message_id_key",
    "method_requires_session_header",
    "session_id_from_message",
    "session_id_from_params",
    "session_id_from_result",
]

# Header names (case-insensitive on the wire; we normalize to these spellings).
CONNECTION_ID_HEADER = "Acp-Connection-Id"
SESSION_ID_HEADER = "Acp-Session-Id"

# MIME types.
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_SSE = "text/event-stream"

# Endpoint path used by docs/examples (the adapter itself is path-agnostic).
ACP_ENDPOINT_PATH = "/acp"

INITIALIZE_METHOD = AGENT_METHODS["initialize"]

# Agent methods that operate on an *already-established* session and therefore
# require the ``Acp-Session-Id`` header on POST + session-scoped routing of their
# response on GET.  Methods that mint/attach a session id (``session/new``,
# ``session/load``, ``session/fork``, ``session/resume``) are deliberately
# excluded: per the RFD their responses come back on the connection-scoped stream
# because the client does not yet have the session-scoped stream open.
# ``session/list`` is connection-level.
_SESSION_SCOPED_METHODS = frozenset({
    AGENT_METHODS["session_set_mode"],
    AGENT_METHODS["session_set_config_option"],
    AGENT_METHODS["session_prompt"],
    AGENT_METHODS["session_cancel"],
    AGENT_METHODS["session_close"],
})


def is_initialize_request(message: dict[str, Any]) -> bool:
    """True if the message is an ``initialize`` JSON-RPC request."""
    return message.get("method") == INITIALIZE_METHOD and "id" in message


def is_response_message(message: dict[str, Any]) -> bool:
    """True if the message is a JSON-RPC response (has ``id`` and no ``method``)."""
    return "id" in message and "method" not in message


def method_requires_session_header(method: str | None) -> bool:
    """True if a POST for ``method`` must carry the ``Acp-Session-Id`` header."""
    return method in _SESSION_SCOPED_METHODS


def message_id_key(message_id: Any) -> str | None:
    """Normalize a JSON-RPC ``id`` (int or str) to a stable string key.

    The Python core assigns integer request ids while the wire may echo them as
    ints or strings; routing tables must key consistently.  Returns ``None`` for
    a missing id.
    """
    if message_id is None:
        return None
    return str(message_id)


def session_id_from_params(params: Any) -> str | None:
    """Extract ``sessionId`` from a request's ``params`` object, if present."""
    if isinstance(params, dict):
        session_id = params.get("sessionId")
        if isinstance(session_id, str):
            return session_id
    return None


def session_id_from_result(result: Any) -> str | None:
    """Extract ``sessionId`` from a response's ``result`` object, if present."""
    if isinstance(result, dict):
        session_id = result.get("sessionId")
        if isinstance(session_id, str):
            return session_id
    return None


def session_id_from_message(message: dict[str, Any]) -> str | None:
    """Extract a ``sessionId`` from either a request's params or a response's result."""
    return session_id_from_params(message.get("params")) or session_id_from_result(message.get("result"))
