from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from acp.exceptions import RequestError

from . import schema
from .meta import PROTOCOL_VERSION

InitializationPhase = Literal["uninitialized", "initializing", "initialized", "failed"]


def read_protocol_version(params: Any) -> int:
    if not isinstance(params, dict):
        raise RequestError.invalid_params({"details": "initialize params must be an object"})
    version = params.get("protocolVersion")
    if isinstance(version, bool) or not isinstance(version, int) or not 0 <= version <= 0xFFFF:
        raise RequestError.invalid_params({"details": "initialize.protocolVersion must be an integer from 0 to 65535"})
    return version


class InitializationState:
    def __init__(self) -> None:
        self._phase: InitializationPhase = "uninitialized"
        self._failure: BaseException | None = None
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._locked = False

    @property
    def phase(self) -> InitializationPhase:
        return self._phase

    def begin(self, request: schema.InitializeRequest) -> None:
        if self._phase != "uninitialized":
            raise RequestError.invalid_request({"details": "ACP v2 connections may only be initialized once"})
        if request.protocol_version != PROTOCOL_VERSION:
            raise RequestError.invalid_params({
                "expectedProtocolVersion": PROTOCOL_VERSION,
                "receivedProtocolVersion": request.protocol_version,
            })
        self._phase = "initializing"

    def complete(self, response: schema.InitializeResponse) -> None:
        if self._phase != "initializing":
            raise RequestError.invalid_request({"details": "ACP v2 initialization is not in progress"})
        if response.protocol_version != PROTOCOL_VERSION:
            raise RequestError.invalid_request({
                "expectedProtocolVersion": PROTOCOL_VERSION,
                "receivedProtocolVersion": response.protocol_version,
            })
        self._phase = "initialized"
        self._failure = None
        self._ready.set()

    def fail(self, error: BaseException) -> None:
        if self._phase == "initialized":
            return
        self._phase = "failed"
        self._failure = error
        self._ready.set()

    async def initialized(self) -> None:
        if self._phase in {"uninitialized", "initializing"}:
            await self._ready.wait()
        if self._phase == "initialized":
            return
        if self._failure is not None:
            raise self._failure
        raise RequestError.invalid_request({"details": "ACP v2 connection has not been initialized"})

    async def initialize_agent(
        self, params: Any, handle: Callable[[dict[str, Any]], Awaitable[schema.InitializeResponse]]
    ) -> schema.InitializeResponse:
        async with self._lock:
            if self._locked:
                raise RequestError.invalid_request({"details": "ACP v2 connections may only be initialized once"})
            requested = read_protocol_version(params)
            if requested < PROTOCOL_VERSION:
                raise RequestError.unsupported_protocol_version(requested, [PROTOCOL_VERSION])
            normalized = {**params, "protocolVersion": PROTOCOL_VERSION}

            was_initialized = self._phase == "initialized"
            self._phase = "initializing"
            self._ready.clear()
            try:
                response = await handle(normalized)
                self.complete(response)
            except BaseException as error:
                if was_initialized:
                    self._phase = "initialized"
                    self._ready.set()
                else:
                    self.fail(error)
                raise
            return response

    async def require(self, method: str, *, lock: bool = False) -> None:
        async with self._lock:
            if self._phase == "initialized":
                if lock:
                    self._locked = True
                return
            if self._phase != "initializing":
                raise RequestError.invalid_request({
                    "details": f"ACP v2 connection must be initialized before {method!r}"
                })
        await self.initialized()
