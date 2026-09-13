"""In-memory cookie store for the WebSocket handshake.

The HTTP client relies on ``httpx``'s built-in cookie jar for session affinity,
but the WebSocket handshake needs a small, explicit store to collect
``Set-Cookie`` headers from the upgrade response and echo them back as a
``Cookie`` request header for the socket lifetime.

This is intentionally minimal: it stores name→value pairs and only honors
expiration attributes that remove a cookie, matching the affinity-only use case
in the RFD.
"""

from __future__ import annotations

__all__ = ["MemoryAcpCookieStore"]


class MemoryAcpCookieStore:
    """A tiny name→value cookie store keyed by cookie name."""

    def __init__(self) -> None:
        self._cookies: dict[str, str] = {}

    def store_set_cookie(self, header_value: str) -> None:
        """Ingest a single ``Set-Cookie`` header value.

        Only the leading ``name=value`` pair is retained; most cookie attributes
        (``; Path=/``, ``; HttpOnly`` etc.) are ignored. Expiration attributes
        that explicitly clear a cookie (``Max-Age=0`` or an epoch ``Expires``
        value) remove any stored cookie with the same name.
        """
        parts = [part.strip() for part in header_value.split(";")]
        first = parts[0]
        if not first or "=" not in first:
            return
        name, _, value = first.partition("=")
        name = name.strip()
        if not name:
            return
        if _is_deletion_cookie(parts[1:]):
            self._cookies.pop(name, None)
            return
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


def _is_deletion_cookie(attributes: list[str]) -> bool:
    for attribute in attributes:
        key, separator, value = attribute.partition("=")
        if not separator:
            continue
        key = key.strip().lower()
        value = value.strip().lower()
        if key == "max-age" and value == "0":
            return True
        if key == "expires" and value in {"thu, 01 jan 1970 00:00:00 gmt", "0"}:
            return True
    return False
