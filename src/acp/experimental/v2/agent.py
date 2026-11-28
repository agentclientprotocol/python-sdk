from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal

from pydantic import AnyUrl, BaseModel

from acp.connection import Connection, MethodHandler
from acp.exceptions import RequestError
from acp.utils import param_model

from . import schema
from ._connection import open_connection
from ._initialization import InitializationState, read_protocol_version
from ._methods import AGENT_REQUESTS_BY_METHOD, CLIENT_REQUESTS_BY_METHOD
from ._params import build_elicitation_request, build_request
from ._router import MethodRouter
from ._schema_base import WIRE_VALIDATION_CONTEXT
from .interfaces import Agent, CreateElicitationRequest, CreateElicitationResponse
from .meta import AGENT_METHODS, CLIENT_METHODS, PROTOCOL_VERSION

__all__ = ["AgentSideConnection", "run_agent"]


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True, exclude_unset=True)


class _AgentRouter:
    def __init__(self, agent: object, state: InitializationState) -> None:
        self._router = MethodRouter(agent, Agent)
        self._state = state
        self._initialize = AGENT_REQUESTS_BY_METHOD[AGENT_METHODS["initialize"]]
        self._lock = asyncio.Lock()
        self._locked = False

    async def __call__(self, method: str, params: Any | None, is_notification: bool) -> Any:
        async with self._lock:
            if not is_notification and method == self._initialize.method:
                if self._locked:
                    raise RequestError.invalid_request({"details": "ACP v2 connections may only be initialized once"})
                requested = read_protocol_version(params)
                if requested < PROTOCOL_VERSION:
                    raise RequestError.unsupported_protocol_version(requested, [PROTOCOL_VERSION])
                normalized = {**params, "protocolVersion": PROTOCOL_VERSION}
                with self._state.attempt():
                    response: schema.InitializeResponse = await self._router.handle_request(
                        self._initialize, normalized
                    )
                    self._state.check_response(response)
                return response
            await self._state.require(method)
            if not is_notification:
                self._locked = True
        return await self._router(method, params, is_notification)


class AgentSideConnection:
    """Strict experimental ACP v2 connection used by an agent."""

    def __init__(
        self,
        agent: object,
        input_stream: Any,
        output_stream: Any = None,
        *,
        _listening: bool = True,
        **connection_kwargs: Any,
    ) -> None:
        self._state = InitializationState()
        router = _AgentRouter(agent, self._state)
        self._conn = open_connection(
            router,
            input_stream,
            output_stream,
            listening=_listening,
            **connection_kwargs,
        )
        if on_connect := getattr(agent, "on_connect", None):
            on_connect(self)

    @classmethod
    def attach(
        cls,
        agent_factory: Callable[[AgentSideConnection], object],
        connection: Connection,
    ) -> tuple[AgentSideConnection, MethodHandler]:
        """Attach an agent-side wrapper to an existing connection."""
        self = cls.__new__(cls)
        self._state = InitializationState()
        self._conn = connection
        agent = agent_factory(self)
        if on_connect := getattr(agent, "on_connect", None):
            on_connect(self)
        router = _AgentRouter(agent, self._state)
        return self, router

    async def _listen(self) -> None:
        await self._conn.main_loop()

    @param_model(schema.RequestPermissionRequest)
    async def request_permission(
        self,
        session_id: str,
        title: str,
        options: list[schema.PermissionOption],
        description: str | None = None,
        subject: schema.ToolCallPermissionSubjectVariant
        | schema.CommandPermissionSubjectVariant
        | schema.OtherPermissionSubject
        | None = None,
        **kwargs: Any,
    ) -> schema.RequestPermissionResponse:
        return await self._request(
            CLIENT_METHODS["session_request_permission"],
            build_request(
                schema.RequestPermissionRequest,
                {
                    "session_id": session_id,
                    "title": title,
                    "options": options,
                    "description": description,
                    "subject": subject,
                },
                kwargs,
            ),
        )

    @param_model(schema.UpdateSessionNotification)
    async def session_update(
        self,
        session_id: str,
        update: schema.UserMessageChunk
        | schema.UserMessageUpdate
        | schema.AgentMessageChunk
        | schema.AgentMessageUpdate
        | schema.AgentThoughtChunk
        | schema.AgentThoughtUpdate
        | schema.ToolCallContentChunkUpdate
        | schema.SessionToolCallUpdate
        | schema.SessionTerminalUpdate
        | schema.SessionTerminalOutputChunk
        | schema.SessionPlanUpdate
        | schema.SessionPlanRemovedUpdate
        | schema.AvailableCommandsUpdate
        | schema.ConfigOptionUpdate
        | schema.SessionInfoUpdate
        | schema.UsageUpdate
        | schema.SessionNotice
        | schema.SessionCompactionUpdate
        | schema.SessionCompactionSummaryChunk
        | schema.SessionReplayComplete
        | schema.SessionPromptSuggestionsUpdate
        | schema.SessionCursorUpdate
        | schema.SessionSubagentUpdate
        | schema.SessionMessageUpdate
        | schema.SessionMessageChunkUpdate
        | schema.OtherSessionUpdate
        | schema.RunningSessionStateUpdate
        | schema.IdleSessionStateUpdate
        | schema.RequiresActionSessionStateUpdate
        | schema.UnknownSessionStateUpdate
        | schema.OtherSessionStateUpdate,
        **kwargs: Any,
    ) -> None:
        await self._notify(
            CLIENT_METHODS["session_update"],
            build_request(schema.UpdateSessionNotification, {"session_id": session_id, "update": update}, kwargs),
        )

    @param_model(schema.MessageMcpRequest)
    async def mcp_message(
        self,
        server_id: str,
        request_id: str,
        method: str,
        params: dict[str, Any] | None = None,
        session_id: str | None = None,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> schema.MessageMcpResponse:
        return await self._request(
            CLIENT_METHODS["mcp_message"],
            build_request(
                schema.MessageMcpRequest,
                {
                    "server_id": server_id,
                    "request_id": request_id,
                    "method": method,
                    "params": params,
                    "session_id": session_id,
                    "tool_call_id": tool_call_id,
                },
                kwargs,
            ),
        )

    @param_model(CreateElicitationRequest)
    async def create_elicitation(
        self,
        message: str,
        mode: str,
        *,
        session_id: str | None = None,
        request_id: int | str | None = None,
        tool_call_id: str | None = None,
        requested_schema: schema.ElicitationSchema | None = None,
        elicitation_id: str | None = None,
        url: str | AnyUrl | None = None,
        **kwargs: Any,
    ) -> CreateElicitationResponse:
        request = build_elicitation_request(
            message=message,
            mode=mode,
            session_id=session_id,
            request_id=request_id,
            tool_call_id=tool_call_id,
            requested_schema=requested_schema,
            elicitation_id=elicitation_id,
            url=url,
            meta=kwargs,
        )
        return await self._request(CLIENT_METHODS["elicitation_create"], request)

    @param_model(schema.CompleteElicitationNotification)
    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        await self._notify(
            CLIENT_METHODS["elicitation_complete"],
            build_request(schema.CompleteElicitationNotification, {"elicitation_id": elicitation_id}, kwargs),
        )

    @param_model(schema.AuthStatusUpdateNotification)
    async def auth_status_update(
        self,
        state: Literal["authenticated"]
        | Literal["unauthenticated"]
        | Literal["not_required"]
        | Literal["unknown"]
        | str,
        accounts: list[schema.AuthAccount] | None = None,
        **kwargs: Any,
    ) -> None:
        await self._notify(
            CLIENT_METHODS["auth_status_update"],
            build_request(schema.AuthStatusUpdateNotification, {"state": state, "accounts": accounts}, kwargs),
        )

    async def send_extension_request(self, method: str, params: Any = None) -> Any:
        await self._state.require(method)
        return await self._conn.send_request(_extension_method(method), params)

    async def send_extension_notification(self, method: str, params: Any = None) -> None:
        await self._state.require(method)
        await self._conn.send_notification(_extension_method(method), params)

    async def close(self) -> None:
        await self._conn.close()

    async def _request(self, method: str, request: BaseModel) -> Any:
        await self._state.require(method)
        spec = CLIENT_REQUESTS_BY_METHOD[method]
        response = await self._conn.send_request(method, _dump(request))
        if response is None and spec.empty_response:
            response = {}
        return spec.response.validate_python(response, context=WIRE_VALIDATION_CONTEXT)

    async def _notify(self, method: str, notification: BaseModel) -> None:
        await self._state.require(method)
        await self._conn.send_notification(method, _dump(notification))

    async def __aenter__(self) -> AgentSideConnection:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()


def _extension_method(method: str) -> str:
    if not method.startswith("_"):
        raise ValueError("ACP extension methods must start with '_'")
    return method


async def run_agent(
    agent: object,
    input_stream: Any = None,
    output_stream: Any = None,
    *,
    stdio_buffer_limit_bytes: int = 50 * 1024 * 1024,
    **connection_kwargs: Any,
) -> None:
    if input_stream is None and output_stream is None:
        from acp.stdio import stdio_streams

        output_stream, input_stream = await stdio_streams(limit=stdio_buffer_limit_bytes)
    connection = AgentSideConnection(
        agent,
        input_stream,
        output_stream,
        _listening=False,
        **connection_kwargs,
    )
    try:
        await connection._listen()
    finally:
        await asyncio.shield(connection.close())
