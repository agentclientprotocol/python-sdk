from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..exceptions import RequestError
from ..interfaces import Agent
from ..meta import AGENT_METHODS
from ..router import MessageRouter, Route, _resolve_handler, _warn_legacy_handler
from ..schema import (
    AuthenticateRequest,
    CancelNotification,
    CloseSessionRequest,
    ForkSessionRequest,
    InitializeRequest,
    ListSessionsRequest,
    LoadSessionRequest,
    NewSessionRequest,
    PromptRequest,
    ResumeSessionRequest,
    SetSessionConfigOptionBooleanRequest,
    SetSessionConfigOptionSelectRequest,
    SetSessionModelRequest,
    SetSessionModeRequest,
)
from ..utils import model_to_kwargs, normalize_result

__all__ = ["build_agent_router"]


_SET_CONFIG_OPTION_MODELS = (SetSessionConfigOptionBooleanRequest, SetSessionConfigOptionSelectRequest)


def _validate_set_config_option_request(params: Any) -> BaseModel:
    if isinstance(params, dict) and params.get("type") == "boolean":
        return SetSessionConfigOptionBooleanRequest.model_validate(params)
    return SetSessionConfigOptionSelectRequest.model_validate(params)


def _make_set_config_option_handler(agent: Agent) -> Any:
    func, attr, legacy_api = _resolve_handler(agent, "set_config_option")
    if func is None:
        return None

    async def wrapper(params: Any) -> Any:
        if legacy_api:
            _warn_legacy_handler(agent, attr)
        request = _validate_set_config_option_request(params)
        if legacy_api:
            return await func(request)
        return await func(**model_to_kwargs(request, _SET_CONFIG_OPTION_MODELS))

    return wrapper


def build_agent_router(agent: Agent, use_unstable_protocol: bool = False) -> MessageRouter:
    router = MessageRouter(use_unstable_protocol=use_unstable_protocol)

    router.route_request(AGENT_METHODS["initialize"], InitializeRequest, agent, "initialize")
    router.route_request(AGENT_METHODS["session_new"], NewSessionRequest, agent, "new_session")
    router.route_request(
        AGENT_METHODS["session_load"],
        LoadSessionRequest,
        agent,
        "load_session",
        adapt_result=normalize_result,
    )
    router.route_request(AGENT_METHODS["session_list"], ListSessionsRequest, agent, "list_sessions")
    router.route_request(
        AGENT_METHODS["session_close"],
        CloseSessionRequest,
        agent,
        "close_session",
        adapt_result=normalize_result,
        unstable=True,
    )
    router.route_request(
        AGENT_METHODS["session_set_mode"],
        SetSessionModeRequest,
        agent,
        "set_session_mode",
        adapt_result=normalize_result,
    )
    router.route_request(AGENT_METHODS["session_prompt"], PromptRequest, agent, "prompt")
    router.route_request(
        AGENT_METHODS["session_set_model"],
        SetSessionModelRequest,
        agent,
        "set_session_model",
        adapt_result=normalize_result,
        unstable=True,
    )
    router.add_route(
        Route(
            method=AGENT_METHODS["session_set_config_option"],
            func=_make_set_config_option_handler(agent),
            kind="request",
            adapt_result=normalize_result,
        )
    )
    router.route_request(
        AGENT_METHODS["authenticate"],
        AuthenticateRequest,
        agent,
        "authenticate",
        adapt_result=normalize_result,
    )
    router.route_request(AGENT_METHODS["session_fork"], ForkSessionRequest, agent, "fork_session", unstable=True)
    router.route_request(AGENT_METHODS["session_resume"], ResumeSessionRequest, agent, "resume_session", unstable=True)

    router.route_notification(AGENT_METHODS["session_cancel"], CancelNotification, agent, "cancel")

    @router.handle_extension_request
    async def _handle_extension_request(name: str, payload: dict[str, Any]) -> Any:
        ext = getattr(agent, "ext_method", None)
        if ext is None:
            raise RequestError.method_not_found(f"_{name}")
        return await ext(name, payload)

    @router.handle_extension_notification
    async def _handle_extension_notification(name: str, payload: dict[str, Any]) -> None:
        ext = getattr(agent, "ext_notification", None)
        if ext is None:
            return
        await ext(name, payload)

    return router
