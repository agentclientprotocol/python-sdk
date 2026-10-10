from __future__ import annotations

import pytest

from acp._cookies import MemoryAcpCookieStore


def test_store_and_render_single_cookie() -> None:
    store = MemoryAcpCookieStore()
    store.store_set_cookie("affinity=abc123; Path=/; HttpOnly")
    assert store.cookie_header() == "affinity=abc123"


def test_store_multiple_cookies_preserves_all() -> None:
    store = MemoryAcpCookieStore()
    store.store_set_cookies(["a=1; Path=/", "b=2; Secure"])
    assert store.cookie_header() == "a=1; b=2"


def test_later_value_overwrites_same_name() -> None:
    store = MemoryAcpCookieStore()
    store.store_set_cookie("a=1")
    store.store_set_cookie("a=2")
    assert store.cookie_header() == "a=2"
    assert len(store) == 1


@pytest.mark.parametrize(
    "expiration_attribute",
    [
        "Max-Age=0",
        "Max-Age=-1",
        "Max-Age=00",
        "Expires=Thu, 01 Jan 1970 00:00:00 GMT",
        "Expires=Thu, 01 Jan 1970 00:00:01 GMT",
        "Expires=Wed, 31 Dec 1969 23:59:59 GMT",
        "Expires=Thu, 01-Jan-1970 00:00:00 GMT",
        "Expires=Mon, 01 Jan 2024 00:00:00 GMT",
    ],
)
def test_expiring_cookie_removes_stored_value(expiration_attribute: str) -> None:
    store = MemoryAcpCookieStore()
    store.store_set_cookie("affinity=abc123; Path=/")
    store.store_set_cookie(f"affinity=; {expiration_attribute}; Path=/")
    assert store.cookie_header() is None
    assert len(store) == 0


def test_positive_max_age_takes_precedence_over_past_expires() -> None:
    store = MemoryAcpCookieStore()
    store.store_set_cookie("affinity=abc123; Path=/")
    store.store_set_cookie("affinity=keepme; Max-Age=3600; Expires=Thu, 01 Jan 1970 00:00:00 GMT")
    assert store.cookie_header() == "affinity=keepme"


def test_empty_store_returns_none() -> None:
    store = MemoryAcpCookieStore()
    assert store.cookie_header() is None


def test_malformed_set_cookie_ignored() -> None:
    store = MemoryAcpCookieStore()
    store.store_set_cookie("garbage")
    store.store_set_cookie("")
    assert store.cookie_header() is None


def test_clear_drops_all() -> None:
    store = MemoryAcpCookieStore()
    store.store_set_cookie("a=1")
    store.clear()
    assert store.cookie_header() is None
