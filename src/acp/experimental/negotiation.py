from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel

from acp import meta as v1_meta
from acp import schema as v1_schema
from acp.agent.connection import AgentSideConnection as V1AgentSideConnection
from acp.connection import Connection, MethodHandler
from acp.exceptions import RequestError
from acp.interfaces import Agent as V1Agent
from acp.interfaces import Client as V1Client

from . import v2
from .v2._connection import open_connection
from .v2.agent import AgentFactory as V2AgentFactory
from .v2.agent import AgentSideConnection as V2AgentSideConnection

__all__ = [
    "AgentProtocolConnection",
    "AgentProtocolRouter",
]

V1AgentFactory = Callable[[V1Client], V1Agent]


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True)


def _read_protocol_version(params: Any) -> int:
    if not isinstance(params, dict):
        raise RequestError.invalid_params({"details": "initialize params must be an object"})
    version = params.get("protocolVersion")
    if isinstance(version, bool) or not isinstance(version, int) or not 0 <= version <= 0xFFFF:
        raise RequestError.invalid_params({"details": "initialize.protocolVersion must be an integer from 0 to 65535"})
    return version


def _v2_initialize_to_v1(request: v2.schema.InitializeRequest) -> v1_schema.InitializeRequest:
    capabilities = (
        v1_schema.ClientCapabilities.model_validate(_dump(request.capabilities))
        if request.capabilities is not None
        else None
    )
    return v1_schema.InitializeRequest(
        protocol_version=v1_meta.PROTOCOL_VERSION,
        client_capabilities=capabilities,
        client_info=v1_schema.Implementation.model_validate(_dump(request.info)),
        field_meta=request.field_meta,
    )


def _normalize_initialize(params: Any, selected_version: int) -> dict[str, Any]:
    requested_version = _read_protocol_version(params)
    if selected_version == v2.PROTOCOL_VERSION:
        request = v2.schema.InitializeRequest.model_validate(params)
        request.protocol_version = v2.PROTOCOL_VERSION
        return _dump(request)
    if requested_version >= v2.PROTOCOL_VERSION:
        request = v2.schema.InitializeRequest.model_validate(params)
        return _dump(_v2_initialize_to_v1(request))
    request = v1_schema.InitializeRequest.model_validate(params)
    request.protocol_version = v1_meta.PROTOCOL_VERSION
    return _dump(request)


class _AgentNegotiationHandler:
    def __init__(
        self,
        v1_agent: V1AgentFactory | V1Agent | None,
        v2_agent: V2AgentFactory | v2.Agent | None,
    ) -> None:
        self._v1_agent = v1_agent
        self._v2_agent = v2_agent
        self._connection: Connection | None = None
        self._selected: MethodHandler | None = None
        self._endpoint: V1AgentSideConnection | V2AgentSideConnection | None = None
        self._lock = asyncio.Lock()

    def bind_connection(self, connection: Connection) -> None:
        self._connection = connection

    async def __call__(self, method: str, params: Any | None, is_notification: bool) -> Any:
        async with self._lock:
            if self._selected is None:
                return await self._initialize(method, params, is_notification)
            if not is_notification and method == v2.AGENT_METHODS["initialize"]:
                raise RequestError.invalid_request({"details": "ACP connections may only be initialized once"})
            handler = self._selected
        return await handler(method, params, is_notification)

    async def _initialize(self, method: str, params: Any, is_notification: bool) -> Any:
        if is_notification or method != v2.AGENT_METHODS["initialize"]:
            raise RequestError.invalid_request({"details": "The first ACP request must be initialize"})
        requested = _read_protocol_version(params)
        selected = self._select(requested)
        connection = self._connection
        if connection is None:
            raise RuntimeError("Protocol router is not connected")

        if selected == v2.PROTOCOL_VERSION:
            endpoint, handler = V2AgentSideConnection._attach(cast(Any, self._v2_agent), connection)
        else:
            endpoint, handler = V1AgentSideConnection._attach(cast(Any, self._v1_agent), connection)
        self._endpoint = endpoint
        self._selected = handler
        normalized = _normalize_initialize(params, selected)
        response = await handler(method, normalized, False)
        parsed_version = _read_protocol_version(_dump(response) if isinstance(response, BaseModel) else response)
        if parsed_version != selected:
            raise RequestError.invalid_request({
                "details": f"initialize response selected protocol {parsed_version}, expected {selected}"
            })
        return response

    def _select(self, requested: int) -> int:
        if self._v2_agent is not None and requested >= v2.PROTOCOL_VERSION:
            return v2.PROTOCOL_VERSION
        if self._v1_agent is not None and requested >= v1_meta.PROTOCOL_VERSION:
            return v1_meta.PROTOCOL_VERSION
        supported = [
            version
            for version, implementation in (
                (v1_meta.PROTOCOL_VERSION, self._v1_agent),
                (v2.PROTOCOL_VERSION, self._v2_agent),
            )
            if implementation is not None
        ]
        raise RequestError.invalid_request({
            "details": f"Unsupported ACP protocol {requested}; configured versions are {supported}"
        })


class AgentProtocolConnection:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    async def listen(self) -> None:
        await self._connection.main_loop()

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
        v1: V1AgentFactory | V1Agent | None = None,
        v2: V2AgentFactory | v2.Agent | None = None,
    ) -> None:
        if v1 is None and v2 is None:
            raise ValueError("Configure at least one ACP protocol implementation")
        self._v1 = v1
        self._v2 = v2

    def connect(
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
        connection = self.connect(
            input_stream,
            output_stream,
            listening=False,
            **connection_kwargs,
        )
        try:
            await connection.listen()
        finally:
            await asyncio.shield(connection.close())
