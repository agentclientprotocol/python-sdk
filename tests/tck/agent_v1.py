"""Deterministic ACP v1 agent for exercising the SDK with the external TCK.

This probe supplies only application state. The SDK owns stdio framing, routing,
model validation, serialization, and JSON-RPC errors.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from uuid import uuid4

from acp import PROTOCOL_VERSION, Agent, InitializeResponse, NewSessionResponse, PromptResponse, run_agent
from acp.interfaces import Client
from acp.schema import AgentCapabilities, AgentMessageChunk, Implementation, TextContentBlock

CANCEL_PROMPT = "wait_for_cancel"


class ProbeAgent:
    """A plain class so unimplemented Protocol methods do not become handlers."""

    def __init__(self) -> None:
        self._client: Client | None = None
        self._pending: dict[str, asyncio.Event] = {}

    def on_connect(self, client: Client) -> None:
        self._client = client

    async def initialize(self, protocol_version: int, **kwargs: Any) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(),
            agent_info=Implementation(name="python-sdk-tck-v1", version="1"),
        )

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        return NewSessionResponse(session_id=uuid4().hex)

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        assert self._client is not None
        wait_for_cancel = (
            len(prompt) == 1 and isinstance(prompt[0], TextContentBlock) and prompt[0].text == CANCEL_PROMPT
        )
        event = asyncio.Event() if wait_for_cancel else None
        if event is not None:
            self._pending[session_id] = event
        try:
            await self._client.session_update(
                session_id=session_id,
                update=AgentMessageChunk(content=TextContentBlock(text="ready")),
            )
            if event is not None:
                await event.wait()
            return PromptResponse(stop_reason="cancelled" if event is not None else "end_turn")
        finally:
            self._pending.pop(session_id, None)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        if event := self._pending.get(session_id):
            event.set()


async def main() -> None:
    # The public Protocol lists optional handlers that this probe intentionally omits.
    await run_agent(cast(Agent, ProbeAgent()))


if __name__ == "__main__":
    asyncio.run(main())
