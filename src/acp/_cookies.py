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

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

__all__ = ["MemoryAcpCookieStore"]


class MemoryAcpCookieStore:
    """A tiny name→value cookie store keyed by cookie name."""

    def __init__(self) -> None:
        self._cookies: dict[str, str] = {}

    def store_set_cookie(self, header_value: str) -> None:
        """Ingest a single ``Set-Cookie`` header value.

        Only the leading ``name=value`` pair is retained; most cookie attributes
        (``; Path=/``, ``; HttpOnly`` etc.) are ignored. Expiration attributes
        that explicitly clear a cookie (non-positive ``Max-Age`` or a past
        ``Expires`` value) remove any stored cookie with the same name.
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
    return _expiry_decision(attributes) is True


def _expiry_decision(attributes: list[str]) -> bool | None:
    """Return whether attributes expire the cookie now.

    ``True`` means delete now, ``False`` means keep the cookie, and ``None``
    means no usable expiration attribute was present. Per RFC 6265 section 5.3,
    a valid ``Max-Age`` attribute takes precedence over ``Expires``.
    """
    expires_verdict: bool | None = None
    for attribute in attributes:
        key, separator, value = attribute.partition("=")
        if not separator:
            continue
        key = key.strip().lower()
        value = value.strip()
        if key == "max-age":
            try:
                return int(value) <= 0
            except ValueError:
                continue
        if key == "expires" and expires_verdict is None:
            try:
                expires_at = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                continue
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            expires_verdict = expires_at <= datetime.now(timezone.utc)
    return expires_verdict
