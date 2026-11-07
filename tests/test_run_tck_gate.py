"""The v2 MANDATORY gate in `scripts/run_tck.py` and its two pinned alpha.5 auth skips.

`scripts/run_tck.py` drives an external TCK checkout, so these tests call `check_mandatory`
directly with synthetic requirement rows instead of running a suite.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.run_tck import V2_PINNED_SKIPS, check_mandatory

NO_AUTH = {"session": {}}
MANDATORY_SKIPS = ["ACP-AUTH-206", "ACP-AUTH-207"]


def row(requirement_id: str, *, status: str = "SKIPPED", reason: str | None = None, nodeid: str | None = None):
    """A MANDATORY row whose single child test is SKIPPED on the pinned node id."""
    pinned_nodeid, pinned_reason = V2_PINNED_SKIPS.get(requirement_id, (nodeid, reason))
    message = repr(("/tck/test_authentication.py", 1, reason if reason is not None else pinned_reason))
    return {
        "id": requirement_id,
        "tier": "MANDATORY",
        "status": status,
        "tests": [{"nodeid": nodeid or pinned_nodeid, "status": "SKIPPED", "message": message}],
    }


def gate(rows: list[dict[str, Any]], version: int = 2, capabilities: Any = NO_AUTH) -> list[str]:
    return check_mandatory(rows, version, {"agent_capabilities": capabilities})


def test_both_pinned_auth_skips_are_accepted_for_an_unauthenticated_v2_fixture() -> None:
    assert gate([row("ACP-AUTH-206"), row("ACP-AUTH-207")]) == []


@pytest.mark.parametrize("requirement_id", MANDATORY_SKIPS)
def test_pin_requires_the_exact_reason_and_node_id(requirement_id: str) -> None:
    assert gate([row(requirement_id, reason=f"{V2_PINNED_SKIPS[requirement_id][1]} (drifted)")]) != []
    assert gate([row(requirement_id, nodeid="test_authentication.py::test_other")]) != []


@pytest.mark.parametrize("capabilities", [None, [], "unexpected", {"session": {}, "auth": {"terminal": {}}}])
def test_pin_requires_a_fixture_that_reports_no_auth(capabilities: Any) -> None:
    assert gate([row("ACP-AUTH-206"), row("ACP-AUTH-207")], capabilities=capabilities) != []


def test_pin_accepts_capabilities_that_omit_auth() -> None:
    assert gate([row("ACP-AUTH-206")], capabilities={}) == []
    assert gate([row("ACP-AUTH-206")], capabilities={"auth": None}) == []


def test_fail_row_with_pinned_id_and_children_is_still_gated() -> None:
    errors = gate([row("ACP-AUTH-206", status="FAIL")])
    assert "v2: ACP-AUTH-206 is FAIL, expected PASS" in errors


def test_unpinned_skip_and_other_lanes_stay_gated() -> None:
    assert gate([row("ACP-AUTH-202", reason="Skipped: anything", nodeid="test_authentication.py::test_other")]) != []
    assert gate([row("ACP-AUTH-206")], version=1) != []


def test_mandatory_failures_are_gated_but_advisory_results_are_not() -> None:
    failing = [
        {"id": "ACP-BATCH-201", "tier": "MANDATORY", "status": "FAIL", "tests": []},
        {"id": "ACP-INIT-201", "tier": "MANDATORY", "status": "NOT_TESTED", "tests": []},
    ]
    assert gate(failing) != []
    advisory = [
        {"id": "ACP-SESSION-004", "tier": "ADVISORY", "status": "SKIPPED", "tests": []},
        {"id": "ACP-AUTH-203", "tier": "CAPABILITY", "status": "SKIPPED", "tests": []},
    ]
    assert gate(advisory) == []
