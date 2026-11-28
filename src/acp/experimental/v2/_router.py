from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from acp.exceptions import RequestError
from acp.utils import model_to_kwargs

from ._methods import NotificationSpec, RequestSpec, protocol_specs
from ._schema_base import WIRE_VALIDATION_CONTEXT

ExtensionRequest = Callable[[str, Any], Awaitable[Any]]
ExtensionNotification = Callable[[str, Any], Awaitable[None]]


class MethodRouter:
    def __init__(
        self,
        target: Any,
        protocol: type,
    ) -> None:
        self._target = target
        self._protocol = protocol
        requests, notifications = protocol_specs(protocol)
        self._requests = {spec.method: spec for spec in requests}
        self._notifications = {spec.method: spec for spec in notifications}

    def request_spec(self, method: str) -> RequestSpec | None:
        return self._requests.get(method)

    def _handler(self, name: str) -> Any:
        handler = getattr(self._target, name, None)
        if getattr(handler, "__func__", handler) is getattr(self._protocol, name, None):
            return None
        return handler

    async def handle_request(self, spec: RequestSpec, params: Any) -> Any:
        handler = self._handler(spec.handler)
        if handler is None:
            raise RequestError.method_not_found(spec.method)
        request = spec.request.validate_python({} if params is None else params, context=WIRE_VALIDATION_CONTEXT)
        response = await handler(**model_to_kwargs(request, type(request)))
        if response is None and spec.empty_response:
            response = {}
        if spec.adapt_result is not None:
            response = spec.adapt_result(response)
        try:
            return spec.response.validate_python(response)
        except ValidationError as exc:
            raise RequestError.internal_error({
                "errors": exc.errors(include_context=False, include_input=False)
            }) from exc

    async def handle_notification(self, spec: NotificationSpec, params: Any) -> None:
        handler = self._handler(spec.handler)
        if handler is None:
            return
        request = spec.params.validate_python({} if params is None else params, context=WIRE_VALIDATION_CONTEXT)
        await handler(**model_to_kwargs(request, type(request)))

    async def __call__(self, method: str, params: Any | None, is_notification: bool) -> Any:
        if method.startswith("_"):
            return await self._handle_extension(method, params, is_notification)
        if is_notification:
            spec = self._notifications.get(method)
            if spec is None:
                return None
            await self.handle_notification(spec, params)
            return None
        spec = self._requests.get(method)
        if spec is None:
            raise RequestError.method_not_found(method)
        return await self.handle_request(spec, params)

    async def _handle_extension(self, method: str, params: Any, is_notification: bool) -> Any:
        handler_name = "handle_extension_notification" if is_notification else "handle_extension_request"
        handler = self._handler(handler_name)
        if handler is None:
            if is_notification:
                return None
            raise RequestError.method_not_found(method)
        return await handler(method, params)
