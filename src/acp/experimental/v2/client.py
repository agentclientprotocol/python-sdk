from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel

from acp.connection import Connection

from . import schema
from ._connection import open_connection
from ._initialization import Initialization, InitializationState
from ._methods import (
    AGENT_REQUESTS_BY_METHOD,
    CLIENT_NOTIFICATIONS,
    CLIENT_REQUESTS,
    SetConfigOptionRequest,
)
from ._router import MethodRouter
from .agent import _dump, _extension_method
from .interfaces import Agent, Client
from .meta import AGENT_METHODS, PROTOCOL_METHODS

__all__ = ["ClientSideConnection", "connect_to_agent"]

ClientFactory = Callable[[Agent], Client]


class _ClientRouter:
    def __init__(self, client: Client, state: InitializationState) -> None:
        self._router = MethodRouter(client, CLIENT_REQUESTS, CLIENT_NOTIFICATIONS)
        self._state = state

    async def __call__(self, method: str, params: Any | None, is_notification: bool) -> Any:
        if is_notification and method == PROTOCOL_METHODS["cancel_request"]:
            if self._state.phase not in {"initializing", "initialized"}:
                await self._state.require(method)
            return None
        await self._state.require(method)
        return await self._router(method, params, is_notification)


class ClientSideConnection:
    """Strict experimental ACP v2 connection used by a client."""

    def __init__(
        self,
        to_client: ClientFactory | Client,
        input_stream: Any,
        output_stream: Any = None,
        **connection_kwargs: Any,
    ) -> None:
        self._state = InitializationState()
        client = to_client(self) if callable(to_client) else to_client
        router = _ClientRouter(cast(Client, client), self._state)
        self._conn = open_connection(router, input_stream, output_stream, **connection_kwargs)
        if on_connect := getattr(client, "on_connect", None):
            on_connect(self)

    @classmethod
    def _attach(
        cls,
        to_client: ClientFactory | Client,
        connection: Connection,
    ) -> tuple[ClientSideConnection, _ClientRouter]:
        self = cls.__new__(cls)
        self._state = InitializationState()
        self._conn = connection
        client = to_client(self) if callable(to_client) else to_client
        router = _ClientRouter(cast(Client, client), self._state)
        if on_connect := getattr(client, "on_connect", None):
            on_connect(self)
        return self, router

    def _complete_initialization(
        self,
        request: schema.InitializeRequest,
        response: schema.InitializeResponse,
    ) -> Initialization:
        self._state.begin(request)
        return self._state.complete(response)

    async def wait_until_initialized(self) -> Initialization:
        return await self._state.initialized()

    async def initialize(self, request: schema.InitializeRequest) -> schema.InitializeResponse:
        self._state.begin(request)
        try:
            response = await self._conn.send_request(AGENT_METHODS["initialize"], _dump(request))
            parsed = schema.InitializeResponse.model_validate(response)
            self._state.complete(parsed)
        except BaseException as error:
            self._state.fail(error)
            await self._conn.close()
            raise
        return parsed

    async def login(self, request: schema.LoginAuthRequest) -> schema.LoginAuthResponse:
        return await self._request(AGENT_METHODS["auth_login"], request)

    async def logout(self, request: schema.LogoutAuthRequest) -> schema.LogoutAuthResponse:
        return await self._request(AGENT_METHODS["auth_logout"], request)

    async def list_providers(self, request: schema.ListProvidersRequest) -> schema.ListProvidersResponse:
        return await self._request(AGENT_METHODS["providers_list"], request)

    async def set_provider(self, request: schema.SetProviderRequest) -> schema.SetProviderResponse:
        return await self._request(AGENT_METHODS["providers_set"], request)

    async def disable_provider(self, request: schema.DisableProviderRequest) -> schema.DisableProviderResponse:
        return await self._request(AGENT_METHODS["providers_disable"], request)

    async def new_session(self, request: schema.NewSessionRequest) -> schema.NewSessionResponse:
        return await self._request(AGENT_METHODS["session_new"], request)

    async def list_sessions(self, request: schema.ListSessionsRequest) -> schema.ListSessionsResponse:
        return await self._request(AGENT_METHODS["session_list"], request)

    async def delete_session(self, request: schema.DeleteSessionRequest) -> schema.DeleteSessionResponse:
        return await self._request(AGENT_METHODS["session_delete"], request)

    async def fork_session(self, request: schema.ForkSessionRequest) -> schema.ForkSessionResponse:
        return await self._request(AGENT_METHODS["session_fork"], request)

    async def resume_session(self, request: schema.ResumeSessionRequest) -> schema.ResumeSessionResponse:
        return await self._request(AGENT_METHODS["session_resume"], request)

    async def close_session(self, request: schema.CloseSessionRequest) -> schema.CloseSessionResponse:
        return await self._request(AGENT_METHODS["session_close"], request)

    async def set_config_option(self, request: SetConfigOptionRequest) -> schema.SetSessionConfigOptionResponse:
        return await self._request(AGENT_METHODS["session_set_config_option"], request)

    async def prompt(self, request: schema.PromptRequest) -> schema.PromptResponse:
        return await self._request(AGENT_METHODS["session_prompt"], request)

    async def cancel(self, notification: schema.CancelSessionNotification) -> None:
        await self._notify(AGENT_METHODS["session_cancel"], notification)

    async def message_mcp(self, message: schema.MessageMcpRequest) -> Any:
        return await self._request(AGENT_METHODS["mcp_message"], message)

    async def notify_mcp(self, notification: schema.MessageMcpNotification) -> None:
        await self._notify(AGENT_METHODS["mcp_message"], notification)

    async def start_nes(self, request: schema.StartNesRequest) -> schema.StartNesResponse:
        return await self._request(AGENT_METHODS["nes_start"], request)

    async def suggest_nes(self, request: schema.SuggestNesRequest) -> schema.SuggestNesResponse:
        return await self._request(AGENT_METHODS["nes_suggest"], request)

    async def accept_nes(self, notification: schema.AcceptNesNotification) -> None:
        await self._notify(AGENT_METHODS["nes_accept"], notification)

    async def reject_nes(self, notification: schema.RejectNesNotification) -> None:
        await self._notify(AGENT_METHODS["nes_reject"], notification)

    async def close_nes(self, request: schema.CloseNesRequest) -> schema.CloseNesResponse:
        return await self._request(AGENT_METHODS["nes_close"], request)

    async def did_open(self, notification: schema.DidOpenDocumentNotification) -> None:
        await self._notify(AGENT_METHODS["document_did_open"], notification)

    async def did_change(self, notification: schema.DidChangeDocumentNotification) -> None:
        await self._notify(AGENT_METHODS["document_did_change"], notification)

    async def did_close(self, notification: schema.DidCloseDocumentNotification) -> None:
        await self._notify(AGENT_METHODS["document_did_close"], notification)

    async def did_save(self, notification: schema.DidSaveDocumentNotification) -> None:
        await self._notify(AGENT_METHODS["document_did_save"], notification)

    async def did_focus(self, notification: schema.DidFocusDocumentNotification) -> None:
        await self._notify(AGENT_METHODS["document_did_focus"], notification)

    async def ext_method(self, method: str, params: Any = None) -> Any:
        await self._state.require(method)
        return await self._conn.send_request(_extension_method(method), params)

    async def ext_notification(self, method: str, params: Any = None) -> None:
        await self._state.require(method)
        await self._conn.send_notification(_extension_method(method), params)

    async def close(self) -> None:
        await self._conn.close()

    async def _request(self, method: str, request: BaseModel) -> Any:
        await self._state.require(method)
        spec = AGENT_REQUESTS_BY_METHOD[method]
        response = await self._conn.send_request(method, _dump(request))
        if response is None and spec.empty_response:
            response = {}
        return spec.response.validate_python(response)

    async def _notify(self, method: str, notification: BaseModel) -> None:
        await self._state.require(method)
        await self._conn.send_notification(method, _dump(notification))

    async def __aenter__(self) -> ClientSideConnection:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()


def connect_to_agent(
    client: ClientFactory | Client,
    input_stream: Any,
    output_stream: Any = None,
    **connection_kwargs: Any,
) -> ClientSideConnection:
    return ClientSideConnection(client, input_stream, output_stream, **connection_kwargs)
