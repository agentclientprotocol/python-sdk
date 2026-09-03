from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from acp.connection import Connection, MethodHandler

from . import schema
from ._connection import open_connection
from ._initialization import InitializationState
from ._methods import (
    AGENT_NOTIFICATIONS,
    AGENT_REQUESTS,
    CLIENT_REQUESTS_BY_METHOD,
    CreateElicitationRequest,
    CreateElicitationResponse,
)
from ._router import MethodRouter
from .meta import CLIENT_METHODS

__all__ = ["AgentSideConnection", "run_agent"]


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True)


class _AgentRouter:
    def __init__(self, agent: object, state: InitializationState) -> None:
        self._router = MethodRouter(agent, AGENT_REQUESTS, AGENT_NOTIFICATIONS)
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

    async def request_permission(
        self,
        request: schema.RequestPermissionRequest,
    ) -> schema.RequestPermissionResponse:
        return await self._request(
            CLIENT_METHODS["session_request_permission"],
            request,
        )

    async def session_update(self, notification: schema.UpdateSessionNotification) -> None:
        await self._notify(CLIENT_METHODS["session_update"], notification)

    async def connect_mcp(self, request: schema.ConnectMcpRequest) -> schema.ConnectMcpResponse:
        return await self._request(CLIENT_METHODS["mcp_connect"], request)

    async def mcp_message(self, message: schema.MessageMcpRequest) -> Any:
        return await self._request(CLIENT_METHODS["mcp_message"], message)

    async def notify_mcp(self, notification: schema.MessageMcpNotification) -> None:
        await self._notify(CLIENT_METHODS["mcp_message"], notification)

    async def disconnect_mcp(
        self,
        request: schema.DisconnectMcpRequest,
    ) -> schema.DisconnectMcpResponse:
        return await self._request(CLIENT_METHODS["mcp_disconnect"], request)

    async def create_elicitation(self, request: CreateElicitationRequest) -> CreateElicitationResponse:
        return await self._request(CLIENT_METHODS["elicitation_create"], request)

    async def complete_elicitation(self, notification: schema.CompleteElicitationNotification) -> None:
        await self._notify(CLIENT_METHODS["elicitation_complete"], notification)

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
