from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import AnyUrl, BaseModel

from acp.connection import Connection, MethodHandler
from acp.utils import param_model

from . import schema
from ._connection import open_connection
from ._initialization import InitializationState
from ._methods import (
    CLIENT_REQUESTS_BY_METHOD,
)
from ._params import build_elicitation_request, build_request
from ._router import MethodRouter
from .interfaces import Agent, CreateElicitationRequest, CreateElicitationResponse
from .meta import CLIENT_METHODS

__all__ = ["AgentSideConnection", "run_agent"]


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True, exclude_unset=True)


class _AgentRouter:
    def __init__(self, agent: object, state: InitializationState) -> None:
        self._router = MethodRouter(agent, Agent)
        self._state = state

    async def __call__(self, method: str, params: Any | None, is_notification: bool) -> Any:
        initialize = self._router.request_spec("initialize")
        if not is_notification and initialize is not None and method == initialize.method:
            request: schema.InitializeRequest = initialize.request.validate_python(params)
            self._state.begin(request)
            try:
                response: schema.InitializeResponse = await self._router.handle_request(initialize, params)
                self._state.complete(response)
            except BaseException as error:
                self._state.fail(error)
                raise
            return response

        await self._state.require(method)
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
        | schema.OtherSessionUpdate
        | schema.RunningSessionStateUpdate
        | schema.IdleSessionStateUpdate
        | schema.RequiresActionSessionStateUpdate
        | schema.OtherSessionStateUpdate,
        **kwargs: Any,
    ) -> None:
        await self._notify(
            CLIENT_METHODS["session_update"],
            build_request(schema.UpdateSessionNotification, {"session_id": session_id, "update": update}, kwargs),
        )

    @param_model(schema.ConnectMcpRequest)
    async def connect_mcp(self, server_id: str, **kwargs: Any) -> schema.ConnectMcpResponse:
        return await self._request(
            CLIENT_METHODS["mcp_connect"], build_request(schema.ConnectMcpRequest, {"server_id": server_id}, kwargs)
        )

    @param_model(schema.MessageMcpRequest)
    async def mcp_message(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        return await self._request(
            CLIENT_METHODS["mcp_message"],
            build_request(
                schema.MessageMcpRequest, {"connection_id": connection_id, "method": method, "params": params}, kwargs
            ),
        )

    @param_model(schema.MessageMcpNotification)
    async def notify_mcp(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        await self._notify(
            CLIENT_METHODS["mcp_message"],
            build_request(
                schema.MessageMcpNotification,
                {"connection_id": connection_id, "method": method, "params": params},
                kwargs,
            ),
        )

    @param_model(schema.DisconnectMcpRequest)
    async def disconnect_mcp(self, connection_id: str, **kwargs: Any) -> schema.DisconnectMcpResponse:
        return await self._request(
            CLIENT_METHODS["mcp_disconnect"],
            build_request(schema.DisconnectMcpRequest, {"connection_id": connection_id}, kwargs),
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
        return spec.response.validate_python(response)

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
