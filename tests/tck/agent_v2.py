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
    AgentMessageUpdate,
    AudioContentBlock,
    CloseSessionResponse,
    EmbeddedResourceContentBlock,
    IdleSessionStateUpdate,
    ImageContentBlock,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    NewSessionResponse,
    OtherContentBlock,
    OtherReplayFrom,
    PromptResponse,
    ReplayFromStartVariant,
    ResourceContentBlock,
    ResumeSessionResponse,
    RunningSessionStateUpdate,
    SessionCapabilities,
    SessionInfo,
    SessionInfoUpdate,
    TextContentBlock,
    UserMessageUpdate,
)

CANCEL_PROMPT = "wait_for_cancel"
_CANCEL_FALLBACK_SECONDS = 60
ContentBlock = (
    TextContentBlock
    | ImageContentBlock
    | AudioContentBlock
    | ResourceContentBlock
    | EmbeddedResourceContentBlock
    | OtherContentBlock
)


class ProbeAgent:
    def __init__(self) -> None:
        self._client: Client | None = None
        self._sessions: dict[str, SessionInfo] = {}
        self._history: dict[str, list[UserMessageUpdate | AgentMessageUpdate]] = {}
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
        self._sessions[session_id] = SessionInfo(session_id=session_id, cwd=cwd, title="Echo")
        self._history[session_id] = []
        return NewSessionResponse(session_id=session_id)

    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        if cursor is not None:
            raise RequestError.invalid_params({"details": "unknown session list cursor"})
        return ListSessionsResponse(sessions=[s for s in self._sessions.values() if cwd is None or s.cwd == cwd])

    def _session(self, session_id: str) -> SessionInfo:
        if session_id not in self._sessions:
            error = RequestError.resource_not_found()
            error.data = {"sessionId": session_id}
            raise error
        return self._sessions[session_id]

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        replay_from: ReplayFromStartVariant | OtherReplayFrom | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        session = self._session(session_id)
        if isinstance(replay_from, OtherReplayFrom):
            raise RequestError.invalid_params({"details": "unsupported replay cursor"})
        assert self._client is not None
        if isinstance(replay_from, ReplayFromStartVariant):
            for message in self._history[session_id]:
                await self._client.session_update(session_id=session_id, update=message)
        await self._client.session_update(session_id=session_id, update=SessionInfoUpdate(title=session.title))
        update = RunningSessionStateUpdate() if session_id in self._cancel_events else IdleSessionStateUpdate()
        await self._client.session_update(session_id=session_id, update=update)
        return ResumeSessionResponse()

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse:
        self._sessions.pop(session_id, None)
        await self.cancel_session(session_id)
        if turn := self._turns.get(session_id):
            await turn
        self._history.pop(session_id, None)
        return CloseSessionResponse()

    async def prompt(self, session_id: str, prompt: list[ContentBlock], **kwargs: Any) -> PromptResponse:
        while turn := self._turns.get(session_id):
            await asyncio.shield(turn)
        self._session(session_id)
        message_id = uuid4().hex
        event = asyncio.Event()
        self._cancel_events[session_id] = event
        self._turns[session_id] = asyncio.create_task(self._run_turn(session_id, message_id, prompt, event))
        return PromptResponse(message_id=message_id)

    async def cancel_session(self, session_id: str, **kwargs: Any) -> None:
        if event := self._cancel_events.get(session_id):
            event.set()

    async def _run_turn(
        self, session_id: str, message_id: str, prompt: list[ContentBlock], event: asyncio.Event
    ) -> None:
        assert self._client is not None
        history = self._history[session_id]
        try:
            message = UserMessageUpdate(message_id=message_id, content=prompt)
            history.append(message)
            await self._client.session_update(session_id=session_id, update=message)
            await self._client.session_update(session_id=session_id, update=RunningSessionStateUpdate())

            texts = [block.text for block in prompt if isinstance(block, TextContentBlock)]
            if texts == [CANCEL_PROMPT]:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(event.wait(), timeout=_CANCEL_FALLBACK_SECONDS)
            elif texts:
                reply_id = uuid4().hex
                for text in texts:
                    await self._client.session_update(
                        session_id=session_id,
                        update=AgentMessageChunk(message_id=reply_id, content=TextContentBlock(text=text)),
                    )
                history.append(AgentMessageUpdate(message_id=reply_id, content=[TextContentBlock(text="".join(texts))]))
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
