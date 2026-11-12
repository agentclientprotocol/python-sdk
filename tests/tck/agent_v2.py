"""Deterministic ACP v2 agent for exercising the experimental SDK runtime.

The session capability includes the v2 lifecycle baseline. Protocol framing,
validation, routing, and errors remain the SDK's responsibility.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from uuid import uuid4

from acp.exceptions import RequestError
from acp.experimental.v2 import PROTOCOL_VERSION, run_agent
from acp.experimental.v2.interfaces import Client
from acp.experimental.v2.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    CloseSessionResponse,
    IdleSessionStateUpdate,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    RunningSessionStateUpdate,
    SessionCapabilities,
    SessionInfo,
    TextContentBlock,
    UserMessageUpdate,
)

CANCEL_PROMPT = "wait_for_cancel"
_CANCEL_FALLBACK_SECONDS = 60


class ProbeAgent:
    def __init__(self) -> None:
        self._client: Client | None = None
        self._sessions: dict[str, str] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._turns: dict[str, asyncio.Task[None]] = {}

    def on_connect(self, client: Client) -> None:
        self._client = client

    async def initialize(self, protocol_version: int, info: Implementation, **kwargs: Any) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            info=Implementation(name="python-sdk-tck-v2", version="1"),
            capabilities=AgentCapabilities(session=SessionCapabilities()),
        )

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        session_id = uuid4().hex
        self._sessions[session_id] = cwd
        return NewSessionResponse(session_id=session_id)

    async def list_sessions(self, cwd: str | None = None, **kwargs: Any) -> ListSessionsResponse:
        sessions = [
            SessionInfo(session_id=session_id, cwd=session_cwd)
            for session_id, session_cwd in self._sessions.items()
            if cwd is None or cwd == session_cwd
        ]
        return ListSessionsResponse(sessions=sessions)

    async def resume_session(
        self, session_id: str, cwd: str, replay_from: Any = None, **kwargs: Any
    ) -> ResumeSessionResponse:
        if replay_from is not None and getattr(replay_from, "type", None) != "start":
            raise RequestError.invalid_params({"details": "unsupported replay cursor"})
        self._sessions[session_id] = cwd
        assert self._client is not None
        update = RunningSessionStateUpdate() if session_id in self._cancel_events else IdleSessionStateUpdate()
        await self._client.session_update(session_id=session_id, update=update)
        return ResumeSessionResponse()

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse:
        self._sessions.pop(session_id, None)
        if event := self._cancel_events.get(session_id):
            event.set()
        if turn := self._turns.get(session_id):
            await turn
        return CloseSessionResponse()

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        message_id = uuid4().hex
        self._turns[session_id] = asyncio.create_task(self._run_turn(session_id, message_id, prompt))
        return PromptResponse(message_id=message_id)

    async def cancel_session(self, session_id: str, **kwargs: Any) -> None:
        if event := self._cancel_events.get(session_id):
            event.set()

    async def _run_turn(self, session_id: str, message_id: str, prompt: list[Any]) -> None:
        assert self._client is not None
        event = asyncio.Event()
        self._cancel_events[session_id] = event
        try:
            await self._client.session_update(
                session_id=session_id,
                update=UserMessageUpdate(message_id=message_id, content=prompt),
            )
            await self._client.session_update(session_id=session_id, update=RunningSessionStateUpdate())

            texts = [block.text for block in prompt if isinstance(block, TextContentBlock)]
            if texts == [CANCEL_PROMPT]:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(event.wait(), timeout=_CANCEL_FALLBACK_SECONDS)
            else:
                for text in texts:
                    await self._client.session_update(
                        session_id=session_id,
                        update=AgentMessageChunk(message_id=message_id, content=TextContentBlock(text=text)),
                    )
            await self._client.session_update(
                session_id=session_id,
                update=IdleSessionStateUpdate(stop_reason="cancelled" if event.is_set() else "end_turn"),
            )
        finally:
            self._cancel_events.pop(session_id, None)
            self._turns.pop(session_id, None)


async def main() -> None:
    await run_agent(ProbeAgent())


if __name__ == "__main__":
    asyncio.run(main())
