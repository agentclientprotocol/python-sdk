from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from ..exceptions import RequestError
from ..interfaces import Client
from ..meta import CLIENT_METHODS
from ..router import MessageRouter, Route, _resolve_handler, _warn_legacy_handler
from ..schema import (
    CompleteElicitationNotification,
    CreateElicitationRequest,
    CreateFormRequestElicitationRequest,
    CreateFormSessionElicitationRequest,
    CreateTerminalRequest,
    CreateUrlRequestElicitationRequest,
    CreateUrlSessionElicitationRequest,
    ElicitationFormRequestMode,
    ElicitationFormSessionMode,
    ElicitationUrlRequestMode,
    ElicitationUrlSessionMode,
    KillTerminalRequest,
    ReadTextFileRequest,
    ReleaseTerminalRequest,
    RequestPermissionRequest,
    SessionNotification,
    TerminalOutputRequest,
    WaitForTerminalExitRequest,
    WriteTextFileRequest,
)
from ..utils import normalize_result

__all__ = ["build_client_router"]
_CREATE_ELICITATION_REQUEST_ADAPTER = TypeAdapter(CreateElicitationRequest)


def _validate_create_elicitation_request(params: Any) -> CreateElicitationRequest:
    return _CREATE_ELICITATION_REQUEST_ADAPTER.validate_python(params)


def _mode_from_create_elicitation_request(
    request: CreateElicitationRequest,
) -> ElicitationFormSessionMode | ElicitationFormRequestMode | ElicitationUrlSessionMode | ElicitationUrlRequestMode:
    if isinstance(request, CreateFormSessionElicitationRequest):
        return ElicitationFormSessionMode(
            session_id=request.session_id,
            tool_call_id=request.tool_call_id,
            requested_schema=request.requested_schema,
        )
    if isinstance(request, CreateFormRequestElicitationRequest):
        return ElicitationFormRequestMode(
            request_id=request.request_id,
            requested_schema=request.requested_schema,
        )

    if isinstance(request, CreateUrlSessionElicitationRequest):
        return ElicitationUrlSessionMode(
            session_id=request.session_id,
            tool_call_id=request.tool_call_id,
            elicitation_id=request.elicitation_id,
            url=request.url,
        )
    if isinstance(request, CreateUrlRequestElicitationRequest):
        return ElicitationUrlRequestMode(
            request_id=request.request_id,
            elicitation_id=request.elicitation_id,
            url=request.url,
        )
    raise RequestError.invalid_params({"details": f"Unsupported elicitation mode: {request.mode!r}"})


def _make_create_elicitation_handler(client: Client) -> Any:
    func, attr, legacy_api = _resolve_handler(client, "create_elicitation")
    if func is None:
        return None

    async def wrapper(params: Any) -> Any:
        if legacy_api:
            _warn_legacy_handler(client, attr)
        request = _validate_create_elicitation_request(params)
        if legacy_api:
            return await func(request)
        kwargs = {"message": request.message, "mode": _mode_from_create_elicitation_request(request)}
        if request.field_meta:
            kwargs.update(request.field_meta)
        return await func(**kwargs)

    return wrapper


def build_client_router(client: Client, use_unstable_protocol: bool = False) -> MessageRouter:
    router = MessageRouter(use_unstable_protocol=use_unstable_protocol)

    router.route_request(CLIENT_METHODS["fs_write_text_file"], WriteTextFileRequest, client, "write_text_file")
    router.route_request(CLIENT_METHODS["fs_read_text_file"], ReadTextFileRequest, client, "read_text_file")
    router.route_request(
        CLIENT_METHODS["session_request_permission"],
        RequestPermissionRequest,
        client,
        "request_permission",
    )
    router.route_request(
        CLIENT_METHODS["terminal_create"],
        CreateTerminalRequest,
        client,
        "create_terminal",
        optional=True,
        default_result=None,
    )
    router.route_request(
        CLIENT_METHODS["terminal_output"],
        TerminalOutputRequest,
        client,
        "terminal_output",
        optional=True,
        default_result=None,
    )
    router.route_request(
        CLIENT_METHODS["terminal_release"],
        ReleaseTerminalRequest,
        client,
        "release_terminal",
        optional=True,
        default_result={},
        adapt_result=normalize_result,
    )
    router.route_request(
        CLIENT_METHODS["terminal_wait_for_exit"],
        WaitForTerminalExitRequest,
        client,
        "wait_for_terminal_exit",
        optional=True,
        default_result=None,
    )
    router.route_request(
        CLIENT_METHODS["terminal_kill"],
        KillTerminalRequest,
        client,
        "kill_terminal",
        optional=True,
        default_result={},
        adapt_result=normalize_result,
    )

    router.add_route(
        Route(
            method=CLIENT_METHODS["elicitation_create"],
            func=_make_create_elicitation_handler(client),
            kind="request",
            adapt_result=normalize_result,
            warn_unstable=not use_unstable_protocol,
        )
    )
    router.route_notification(
        CLIENT_METHODS["elicitation_complete"],
        CompleteElicitationNotification,
        client,
        "complete_elicitation",
        unstable=True,
    )

    router.route_notification(CLIENT_METHODS["session_update"], SessionNotification, client, "session_update")

    @router.handle_extension_request
    async def _handle_extension_request(name: str, payload: dict[str, Any]) -> Any:
        ext = getattr(client, "ext_method", None)
        if ext is None:
            raise RequestError.method_not_found(f"_{name}")
        return await ext(name, payload)

    @router.handle_extension_notification
    async def _handle_extension_notification(name: str, payload: dict[str, Any]) -> None:
        ext = getattr(client, "ext_notification", None)
        if ext is None:
            return
        await ext(name, payload)

    return router
