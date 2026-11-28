from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from acp import meta as v1_meta
from acp import schema as v1_schema
from acp.agent.connection import AgentSideConnection as V1AgentSideConnection
from acp.connection import Connection, MethodHandler
from acp.exceptions import RequestError
from acp.interfaces import Agent as V1Agent

from . import v2
from .v2._connection import open_connection
from .v2._initialization import read_protocol_version
from .v2.agent import AgentSideConnection as V2AgentSideConnection
from .v2.meta import AGENT_METHODS as V2_AGENT_METHODS

__all__ = [
    "AgentProtocolConnection",
    "AgentProtocolRouter",
]

V1AgentFactory = Callable[[V1AgentSideConnection], V1Agent]
V2AgentFactory = Callable[[V2AgentSideConnection], object]


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True)


def _v2_initialize_to_v1(request: v2.schema.InitializeRequest) -> v1_schema.InitializeRequest:
    capabilities = None
    if request.capabilities is not None:
        raw_caps = _dump(request.capabilities)
        auth = request.capabilities.auth
        if auth is not None and auth.terminal is not None:
            raw_caps["auth"]["terminal"] = True
        capabilities = v1_schema.ClientCapabilities.model_validate(raw_caps)
    return v1_schema.InitializeRequest(
        protocol_version=v1_meta.PROTOCOL_VERSION,
        client_capabilities=capabilities,
        client_info=v1_schema.Implementation.model_validate(_dump(request.info)),
        field_meta=request.field_meta,
    )


def _normalize_v1_initialize(params: Any, requested_version: int) -> dict[str, Any]:
    if requested_version >= v2.PROTOCOL_VERSION:
        request = v2.schema.InitializeRequest.model_validate(params)
        return _dump(_v2_initialize_to_v1(request))
    request = v1_schema.InitializeRequest.model_validate(params)
    request.protocol_version = v1_meta.PROTOCOL_VERSION
    return _dump(request)


class _AgentNegotiationHandler:
    def __init__(
        self,
        v1_agent: V1AgentFactory | None,
        v2_agent: V2AgentFactory | None,
    ) -> None:
        self._v1_agent = v1_agent
        self._v2_agent = v2_agent
        self._connection: Connection | None = None
        self._selected: MethodHandler | None = None
        self._selected_version: int | None = None
        self._locked = False
        self._lock = asyncio.Lock()

    def bind_connection(self, connection: Connection) -> None:
        self._connection = connection

    async def __call__(self, method: str, params: Any | None, is_notification: bool) -> Any:
        async with self._lock:
            if not is_notification and method == V2_AGENT_METHODS["initialize"]:
                if self._locked:
                    raise RequestError.invalid_request({"details": "ACP connections may only be initialized once"})
                return await self._initialize(method, params)
            handler = self._selected
            if handler is None:
                raise RequestError.invalid_request({"details": "The first ACP request must be initialize"})
            if not is_notification:
                self._locked = True
        return await handler(method, params, is_notification)

    async def _initialize(self, method: str, params: Any) -> Any:
        requested = read_protocol_version(params)
        connection = self._connection
        if connection is None:
            raise RuntimeError("Protocol router is not connected")

        if self._v2_agent is not None and requested >= v2.PROTOCOL_VERSION:
            selected = v2.PROTOCOL_VERSION
            handler = (
                self._selected
                if self._selected is not None and self._selected_version == selected
                else V2AgentSideConnection.attach(self._v2_agent, connection)[1]
            )
        elif self._v1_agent is not None and requested >= v1_meta.PROTOCOL_VERSION:
            selected = v1_meta.PROTOCOL_VERSION
            handler = (
                self._selected
                if self._selected is not None and self._selected_version == selected
                else V1AgentSideConnection.attach(self._v1_agent, connection)[1]
            )
        else:
            supported = [
                version
                for version, implementation in (
                    (v1_meta.PROTOCOL_VERSION, self._v1_agent),
                    (v2.PROTOCOL_VERSION, self._v2_agent),
                )
                if implementation is not None
            ]
            raise RequestError.unsupported_protocol_version(requested, supported)
        if selected == v1_meta.PROTOCOL_VERSION:
            params = _normalize_v1_initialize(params, requested)
        response = await handler(method, params, False)
        parsed_version = (
            response.protocol_version
            if isinstance(response, v2.schema.InitializeResponse)
            else read_protocol_version(_dump(response) if isinstance(response, BaseModel) else response)
        )
        if parsed_version != selected:
            raise RequestError.invalid_request({
                "details": f"initialize response selected protocol {parsed_version}, expected {selected}"
            })
        self._selected = handler
        self._selected_version = selected
        return response


class AgentProtocolConnection:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    async def _listen(self) -> None:
        await self._connection.main_loop()

    async def listen(self) -> None:
        await self._listen()

    async def close(self) -> None:
        await self._connection.close()

    async def __aenter__(self) -> AgentProtocolConnection:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()


class AgentProtocolRouter:
    """Select one strict agent runtime from the first initialize request."""

    def __init__(
        self,
        *,
        v1: V1AgentFactory | None = None,
        v2: V2AgentFactory | None = None,
    ) -> None:
        if v1 is None and v2 is None:
            raise ValueError("Configure at least one ACP protocol implementation")
        self._v1 = v1
        self._v2 = v2

    def connect(
        self,
        input_stream: Any,
        output_stream: Any = None,
        **connection_kwargs: Any,
    ) -> AgentProtocolConnection:
        return self._connect(input_stream, output_stream, **connection_kwargs)

    def _connect(
        self,
        input_stream: Any,
        output_stream: Any = None,
        *,
        listening: bool = True,
        **connection_kwargs: Any,
    ) -> AgentProtocolConnection:
        handler = _AgentNegotiationHandler(self._v1, self._v2)
        connection = open_connection(
            handler,
            input_stream,
            output_stream,
            listening=listening,
            **connection_kwargs,
        )
        handler.bind_connection(connection)
        return AgentProtocolConnection(connection)

    async def run(
        self,
        input_stream: Any = None,
        output_stream: Any = None,
        *,
        stdio_buffer_limit_bytes: int = 50 * 1024 * 1024,
        **connection_kwargs: Any,
    ) -> None:
        if input_stream is None and output_stream is None:
            from acp.stdio import stdio_streams

            output_stream, input_stream = await stdio_streams(limit=stdio_buffer_limit_bytes)
        connection = self._connect(
            input_stream,
            output_stream,
            listening=False,
            **connection_kwargs,
        )
        try:
            await connection._listen()
        finally:
            await asyncio.shield(connection.close())
