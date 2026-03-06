"""Tests for InMemoryMessageStateStore — outgoing/incoming message state tracking."""

from __future__ import annotations

import pytest

from acp.task.state import InMemoryMessageStateStore


@pytest.fixture
def store() -> InMemoryMessageStateStore:
    return InMemoryMessageStateStore()


@pytest.mark.asyncio
async def test_register_and_resolve_outgoing(store: InMemoryMessageStateStore) -> None:
    """Resolving an outgoing request should fulfill its future."""
    future = store.register_outgoing(1, "session/initialize")
    store.resolve_outgoing(1, {"ok": True})
    result = await future
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_reject_outgoing(store: InMemoryMessageStateStore) -> None:
    """Rejecting an outgoing request should set an exception on its future."""
    future = store.register_outgoing(2, "session/update")
    error = Exception("server error")
    store.reject_outgoing(2, error)
    with pytest.raises(Exception, match="server error"):
        await future


@pytest.mark.asyncio
async def test_resolve_unknown_id_is_noop(store: InMemoryMessageStateStore) -> None:
    """Resolving a non-existent request ID should not raise."""
    store.resolve_outgoing(999, "ignored")


@pytest.mark.asyncio
async def test_reject_unknown_id_is_noop(store: InMemoryMessageStateStore) -> None:
    """Rejecting a non-existent request ID should not raise."""
    store.reject_outgoing(999, Exception("ignored"))


@pytest.mark.asyncio
async def test_reject_all_outgoing(store: InMemoryMessageStateStore) -> None:
    """reject_all_outgoing should fail every pending future."""
    f1 = store.register_outgoing(1, "m1")
    f2 = store.register_outgoing(2, "m2")
    f3 = store.register_outgoing(3, "m3")

    store.reject_all_outgoing(ConnectionError("disconnected"))

    for future in [f1, f2, f3]:
        with pytest.raises(ConnectionError, match="disconnected"):
            await future


@pytest.mark.asyncio
async def test_reject_all_skips_already_done(store: InMemoryMessageStateStore) -> None:
    """reject_all should skip futures that are already resolved."""
    f1 = store.register_outgoing(1, "m1")
    store.resolve_outgoing(1, "done")

    f2 = store.register_outgoing(2, "m2")

    store.reject_all_outgoing(ConnectionError("dc"))

    assert await f1 == "done"
    with pytest.raises(ConnectionError):
        await f2


def test_begin_incoming(store: InMemoryMessageStateStore) -> None:
    """begin_incoming should create a pending record."""
    record = store.begin_incoming("task/update", {"id": "t1"})
    assert record.method == "task/update"
    assert record.params == {"id": "t1"}
    assert record.status == "pending"


def test_complete_incoming(store: InMemoryMessageStateStore) -> None:
    """complete_incoming should update the record status and result."""
    record = store.begin_incoming("task/update", {})
    store.complete_incoming(record, {"success": True})
    assert record.status == "completed"
    assert record.result == {"success": True}


def test_fail_incoming(store: InMemoryMessageStateStore) -> None:
    """fail_incoming should update the record status and error."""
    record = store.begin_incoming("task/update", {})
    store.fail_incoming(record, {"code": -32603})
    assert record.status == "failed"
    assert record.error == {"code": -32603}
