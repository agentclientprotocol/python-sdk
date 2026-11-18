"""Session setup requests carry `systemPrompt` to the agent."""

import asyncio
import contextlib
from typing import Any

import pytest

from acp import schema


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("new_session", {"cwd": "/w", "mcp_servers": []}),
        ("load_session", {"cwd": "/w", "session_id": "s", "mcp_servers": []}),
        ("resume_session", {"cwd": "/w", "session_id": "s", "mcp_servers": []}),
        ("fork_session", {"cwd": "/w", "session_id": "s", "mcp_servers": []}),
    ],
)
async def test_system_prompt_reaches_the_agent(connect, agent, name, params):
    received: list[dict[str, Any]] = []
    done = asyncio.Event()
    original = getattr(agent, name, None)

    async def handler(**kwargs: Any) -> Any:
        received.append(kwargs)
        done.set()
        return await original(**kwargs) if original else None

    setattr(agent, name, handler)
    _, connection = connect(use_unstable_protocol=True)
    prompt = schema.SystemPrompt(mode="append", content=[schema.TextContentBlock(type="text", text="be brief")])
    # The test agent may answer with an error; only what it received matters here.
    with contextlib.suppress(Exception):
        await getattr(connection, name)(**params, system_prompt=prompt)
    await asyncio.wait_for(done.wait(), 2)
    assert received[0]["system_prompt"] == prompt
