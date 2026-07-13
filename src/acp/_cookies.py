"""In-memory cookie store for the WebSocket handshake.

The HTTP client relies on ``httpx``'s built-in cookie jar for session affinity,
but the WebSocket handshake needs a small, explicit store to collect
``Set-Cookie`` headers from the upgrade response and echo them back as a
``Cookie`` request header for the socket lifetime.

This is intentionally minimal: it stores name→value pairs without attribute
parsing (domain/path/expiry), matching the affinity-only use case in the RFD.
"""

from __future__ import annotations

__all__ = ["MemoryAcpCookieStore"]


class MemoryAcpCookieStore:
    """A tiny name→value cookie store keyed by cookie name."""

    def __init__(self) -> None:
        self._cookies: dict[str, str] = {}

    def store_set_cookie(self, header_value: str) -> None:
        """Ingest a single ``Set-Cookie`` header value.

        Only the leading ``name=value`` pair is retained; cookie attributes
        (``; Path=/``, ``; HttpOnly`` etc.) are ignored.
        """
        first = header_value.split(";", 1)[0].strip()
        if not first or "=" not in first:
            return
        name, _, value = first.partition("=")
        name = name.strip()
        if name:
            self._cookies[name] = value.strip()

    def store_set_cookies(self, header_values: list[str]) -> None:
        """Ingest multiple ``Set-Cookie`` header values."""
        for value in header_values:
            self.store_set_cookie(value)

    def cookie_header(self) -> str | None:
        """Render the stored cookies as a ``Cookie`` request header value."""
        if not self._cookies:
            return None
        return "; ".join(f"{name}={value}" for name, value in self._cookies.items())

    def clear(self) -> None:
        """Drop all stored cookies."""
        self._cookies.clear()

    def __len__(self) -> int:
        return len(self._cookies)
