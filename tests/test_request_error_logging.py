import asyncio
import json
import logging
from typing import cast

import pytest

from acp import Agent
from acp.core import AgentSideConnection
from tests.conftest import TestAgent


@pytest.mark.asyncio
async def test_unexpected_handler_error_is_logged_and_returns_internal_error(server, caplog):
    class _RaisingAgent(TestAgent):
        __test__ = False

        async def initialize(
            self,
            protocol_version,
            client_capabilities=None,
            client_info=None,
            **kwargs,
        ):
            raise RuntimeError("boom")

    AgentSideConnection(
        cast(Agent, _RaisingAgent()),
        server.server_writer,
        server.server_reader,
        listening=True,
    )

    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1}}
    with caplog.at_level(logging.ERROR):
        server.client_writer.write((json.dumps(req) + "\n").encode())
        await server.client_writer.drain()
        line = await asyncio.wait_for(server.client_reader.readline(), timeout=1)

    resp = json.loads(line)
    assert resp["id"] == 1
    assert resp["error"]["code"] == -32603  # internal error

    # The original exception (with its traceback) must be logged, not silently
    # swallowed into the JSON-RPC response.
    assert any(record.exc_info for record in caplog.records)
