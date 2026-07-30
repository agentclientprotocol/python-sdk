from __future__ import annotations

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
