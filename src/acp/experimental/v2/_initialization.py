from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from acp.exceptions import RequestError

from . import schema
from .meta import PROTOCOL_VERSION


def read_protocol_version(params: Any) -> int:
    if not isinstance(params, dict):
        raise RequestError.invalid_params({"details": "initialize params must be an object"})
    version = params.get("protocolVersion")
    if isinstance(version, bool) or not isinstance(version, int) or not 0 <= version <= 0xFFFF:
        raise RequestError.invalid_params({"details": "initialize.protocolVersion must be an integer from 0 to 65535"})
    return version


class InitializationState:
    def __init__(self) -> None:
        self._initialized = False
        self._client_started = False
        self._attempt: asyncio.Future[BaseException | None] | None = None

    def begin_client(self, request: schema.InitializeRequest) -> None:
        if self._client_started:
            raise RequestError.invalid_request({"details": "ACP v2 connections may only be initialized once"})
        if request.protocol_version != PROTOCOL_VERSION:
            raise RequestError.invalid_params({
                "expectedProtocolVersion": PROTOCOL_VERSION,
                "receivedProtocolVersion": request.protocol_version,
            })

        self._client_started = True

    @contextmanager
    def attempt(self) -> Iterator[None]:
        attempt = self._attempt = asyncio.get_running_loop().create_future()
        try:
            yield
        except BaseException as error:
            attempt.set_result(error)
            raise
        else:
            self._initialized = True
            attempt.set_result(None)
        finally:
            self._attempt = None

    @staticmethod
    def check_response(response: schema.InitializeResponse) -> None:
        if response.protocol_version != PROTOCOL_VERSION:
            raise RequestError.invalid_request({
                "expectedProtocolVersion": PROTOCOL_VERSION,
                "receivedProtocolVersion": response.protocol_version,
            })

    async def require(self, method: str) -> None:
        if self._initialized:
            return
        attempt = self._attempt
        if attempt is None:
            raise RequestError.invalid_request({"details": f"ACP v2 connection must be initialized before {method!r}"})
        if (error := await asyncio.shield(attempt)) is not None:
            raise error
