from __future__ import annotations

import asyncio
from typing import Literal

from acp.exceptions import RequestError

from . import schema
from .meta import PROTOCOL_VERSION

InitializationPhase = Literal["uninitialized", "initializing", "initialized", "failed"]


class InitializationState:
    def __init__(self) -> None:
        self._phase: InitializationPhase = "uninitialized"
        self._failure: BaseException | None = None
        self._ready = asyncio.Event()

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

    async def require(self, method: str) -> None:
        if self._phase == "initialized":
            return
        if self._phase == "initializing":
            await self.initialized()
            return
        raise RequestError.invalid_request({"details": f"ACP v2 connection must be initialized before {method!r}"})
