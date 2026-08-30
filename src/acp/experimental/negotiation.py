from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel

from acp import meta as v1_meta
from acp import schema as v1_schema
from acp.agent.connection import AgentSideConnection as V1AgentSideConnection
from acp.client.connection import ClientSideConnection as V1ClientSideConnection
from acp.connection import Connection, MethodHandler
from acp.exceptions import RequestError
from acp.interfaces import Agent as V1Agent
from acp.interfaces import Client as V1Client

from . import v2
from .v2._connection import open_connection
from .v2.agent import AgentFactory as V2AgentFactory
from .v2.agent import AgentSideConnection as V2AgentSideConnection
from .v2.client import ClientFactory as V2ClientFactory
from .v2.client import ClientSideConnection as V2ClientSideConnection

__all__ = [
    "AgentProtocolConnection",
    "AgentProtocolRouter",
    "ClientNegotiator",
    "NegotiatedClient",
    "NegotiatedV1",
    "NegotiatedV2",
    "UnsupportedProtocolVersionError",
    "V1ClientConfig",
    "V2ClientConfig",
]

V1AgentFactory = Callable[[V1Client], V1Agent]
V1ClientFactory = Callable[[V1Agent], V1Client]


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


class _SwitchingHandler:
    def __init__(self) -> None:
        self._handler: MethodHandler | None = None
        self._failure: BaseException | None = None
        self._ready = asyncio.Event()

    def bind(self, handler: MethodHandler) -> None:
        if self._handler is not None or self._failure is not None:
            raise RuntimeError("Protocol handler has already been resolved")
        self._handler = handler
        self._ready.set()

    def fail(self, error: BaseException) -> None:
        if self._handler is not None:
            return
        self._failure = error
        self._ready.set()

    async def __call__(self, method: str, params: Any | None, is_notification: bool) -> Any:
        await self._ready.wait()
        if self._failure is not None:
            raise self._failure
        if self._handler is None:
            raise RuntimeError("Protocol handler was not resolved")
        return await self._handler(method, params, is_notification)


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


@dataclass(frozen=True, slots=True)
class V1ClientConfig:
    client: V1ClientFactory | V1Client
    initialize: v1_schema.InitializeRequest

    def __post_init__(self) -> None:
        if self.initialize.protocol_version != v1_meta.PROTOCOL_VERSION:
            raise ValueError(f"V1ClientConfig requires protocol version {v1_meta.PROTOCOL_VERSION}")


@dataclass(frozen=True, slots=True)
class V2ClientConfig:
    client: V2ClientFactory | v2.Client
    initialize: v2.schema.InitializeRequest

    def __post_init__(self) -> None:
        if self.initialize.protocol_version != v2.PROTOCOL_VERSION:
            raise ValueError(f"V2ClientConfig requires protocol version {v2.PROTOCOL_VERSION}")


@dataclass(frozen=True, slots=True)
class NegotiatedV1:
    connection: V1ClientSideConnection
    initialize: v1_schema.InitializeResponse
    protocol_version: int = v1_meta.PROTOCOL_VERSION


@dataclass(frozen=True, slots=True)
class NegotiatedV2:
    connection: V2ClientSideConnection
    initialize: v2.schema.InitializeResponse
    protocol_version: int = v2.PROTOCOL_VERSION


NegotiatedClient = NegotiatedV1 | NegotiatedV2


class UnsupportedProtocolVersionError(ValueError):
    def __init__(self, requested: int, offered: int, supported: frozenset[int]) -> None:
        self.requested = requested
        self.offered = offered
        self.supported = supported
        super().__init__(f"Agent selected ACP protocol {offered}; requested {requested}, supported {sorted(supported)}")


class ClientNegotiator:
    """Send one initialize request and return the selected typed client."""

    def __init__(
        self,
        input_stream: Any,
        output_stream: Any = None,
        *,
        v1: V1ClientConfig | None = None,
        v2: V2ClientConfig | None = None,
        **connection_kwargs: Any,
    ) -> None:
        if v1 is None and v2 is None:
            raise ValueError("Configure at least one ACP client version")
        self._v1 = v1
        self._v2 = v2
        self._handler = _SwitchingHandler()
        self._connection = open_connection(
            self._handler,
            input_stream,
            output_stream,
            **connection_kwargs,
        )
        self._lock = asyncio.Lock()
        self._resolved: NegotiatedClient | None = None
        self._failure: BaseException | None = None

    async def negotiate(self) -> NegotiatedClient:
        async with self._lock:
            if self._resolved is not None:
                return self._resolved
            if self._failure is not None:
                raise self._failure
            try:
                self._resolved = await self._negotiate_once()
            except BaseException as error:
                self._failure = error
                self._handler.fail(error)
                await self._connection.close()
                raise
            return self._resolved

    async def _negotiate_once(self) -> NegotiatedClient:
        offered_request: BaseModel = (
            self._v2.initialize if self._v2 is not None else cast(V1ClientConfig, self._v1).initialize
        )

        response = await self._connection.send_request(v2.AGENT_METHODS["initialize"], _dump(offered_request))
        offered = _read_protocol_version(response)
        requested = offered_request.protocol_version

        if offered == v2.PROTOCOL_VERSION and self._v2 is not None:
            initialize = v2.schema.InitializeResponse.model_validate(response)
            connection, handler = V2ClientSideConnection._attach(self._v2.client, self._connection)
            connection._complete_initialization(self._v2.initialize, initialize)
            self._handler.bind(handler)
            return NegotiatedV2(connection, initialize)
        if offered == v1_meta.PROTOCOL_VERSION and self._v1 is not None:
            initialize = v1_schema.InitializeResponse.model_validate(response)
            connection, handler = V1ClientSideConnection._attach(self._v1.client, self._connection)
            self._handler.bind(handler)
            return NegotiatedV1(connection, initialize)
        supported = frozenset(
            version
            for version, config in (
                (v1_meta.PROTOCOL_VERSION, self._v1),
                (v2.PROTOCOL_VERSION, self._v2),
            )
            if config is not None
        )
        raise UnsupportedProtocolVersionError(requested, offered, supported)

    async def close(self) -> None:
        await self._connection.close()

    async def __aenter__(self) -> ClientNegotiator:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()
