from __future__ import annotations

from typing import Any, Literal

from pydantic import AnyUrl, BaseModel

from acp.utils import param_model

from . import schema
from ._connection import open_connection
from ._initialization import InitializationState
from ._methods import (
    AGENT_REQUESTS_BY_METHOD,
)
from ._params import build_config_request, build_request
from ._router import MethodRouter
from .agent import _dump, _extension_method
from .interfaces import Client, SetConfigOptionRequest
from .meta import AGENT_METHODS

__all__ = ["ClientSideConnection", "connect_to_agent"]


class _ClientRouter:
    def __init__(self, client: object, state: InitializationState) -> None:
        self._router = MethodRouter(client, Client)
        self._state = state

    async def __call__(self, method: str, params: Any | None, is_notification: bool) -> Any:
        await self._state.require(method)
        return await self._router(method, params, is_notification)


class ClientSideConnection:
    """Strict experimental ACP v2 connection used by a client."""

    def __init__(
        self,
        client: object,
        input_stream: Any,
        output_stream: Any = None,
        **connection_kwargs: Any,
    ) -> None:
        self._state = InitializationState()
        router = _ClientRouter(client, self._state)
        self._conn = open_connection(router, input_stream, output_stream, **connection_kwargs)
        if on_connect := getattr(client, "on_connect", None):
            on_connect(self)

    @param_model(schema.InitializeRequest)
    async def initialize(
        self,
        protocol_version: int,
        info: schema.Implementation,
        capabilities: schema.ClientCapabilities | None = None,
        **kwargs: Any,
    ) -> schema.InitializeResponse:
        request = build_request(
            schema.InitializeRequest,
            {"protocol_version": protocol_version, "info": info, "capabilities": capabilities},
            kwargs,
        )
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

    @param_model(schema.LoginAuthRequest)
    async def login(self, method_id: str, **kwargs: Any) -> schema.LoginAuthResponse:
        return await self._request(
            AGENT_METHODS["auth_login"], build_request(schema.LoginAuthRequest, {"method_id": method_id}, kwargs)
        )

    @param_model(schema.LogoutAuthRequest)
    async def logout(self, **kwargs: Any) -> schema.LogoutAuthResponse:
        return await self._request(AGENT_METHODS["auth_logout"], build_request(schema.LogoutAuthRequest, {}, kwargs))

    @param_model(schema.ListProvidersRequest)
    async def list_providers(self, **kwargs: Any) -> schema.ListProvidersResponse:
        return await self._request(
            AGENT_METHODS["providers_list"], build_request(schema.ListProvidersRequest, {}, kwargs)
        )

    @param_model(schema.SetProviderRequest)
    async def set_provider(
        self,
        provider_id: str,
        api_type: Literal["anthropic"]
        | Literal["openai"]
        | Literal["azure"]
        | Literal["vertex"]
        | Literal["bedrock"]
        | str,
        base_url: str | AnyUrl,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> schema.SetProviderResponse:
        return await self._request(
            AGENT_METHODS["providers_set"],
            build_request(
                schema.SetProviderRequest,
                {"provider_id": provider_id, "api_type": api_type, "base_url": base_url, "headers": headers},
                kwargs,
            ),
        )

    @param_model(schema.DisableProviderRequest)
    async def disable_provider(self, provider_id: str, **kwargs: Any) -> schema.DisableProviderResponse:
        return await self._request(
            AGENT_METHODS["providers_disable"],
            build_request(schema.DisableProviderRequest, {"provider_id": provider_id}, kwargs),
        )

    @param_model(schema.NewSessionRequest)
    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[schema.HttpMcpServer | schema.AcpMcpServer | schema.StdioMcpServer | schema.OtherMcpServer]
        | None = None,
        **kwargs: Any,
    ) -> schema.NewSessionResponse:
        return await self._request(
            AGENT_METHODS["session_new"],
            build_request(
                schema.NewSessionRequest,
                {"cwd": cwd, "additional_directories": additional_directories, "mcp_servers": mcp_servers},
                kwargs,
            ),
        )

    @param_model(schema.ListSessionsRequest)
    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
    ) -> schema.ListSessionsResponse:
        return await self._request(
            AGENT_METHODS["session_list"],
            build_request(schema.ListSessionsRequest, {"cwd": cwd, "cursor": cursor}, kwargs),
        )

    @param_model(schema.DeleteSessionRequest)
    async def delete_session(self, session_id: str, **kwargs: Any) -> schema.DeleteSessionResponse:
        return await self._request(
            AGENT_METHODS["session_delete"],
            build_request(schema.DeleteSessionRequest, {"session_id": session_id}, kwargs),
        )

    @param_model(schema.ForkSessionRequest)
    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[schema.HttpMcpServer | schema.AcpMcpServer | schema.StdioMcpServer | schema.OtherMcpServer]
        | None = None,
        **kwargs: Any,
    ) -> schema.ForkSessionResponse:
        return await self._request(
            AGENT_METHODS["session_fork"],
            build_request(
                schema.ForkSessionRequest,
                {
                    "session_id": session_id,
                    "cwd": cwd,
                    "additional_directories": additional_directories,
                    "mcp_servers": mcp_servers,
                },
                kwargs,
            ),
        )

    @param_model(schema.ResumeSessionRequest)
    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[schema.HttpMcpServer | schema.AcpMcpServer | schema.StdioMcpServer | schema.OtherMcpServer]
        | None = None,
        replay_from: schema.ReplayFromStartVariant | schema.OtherReplayFrom | None = None,
        **kwargs: Any,
    ) -> schema.ResumeSessionResponse:
        return await self._request(
            AGENT_METHODS["session_resume"],
            build_request(
                schema.ResumeSessionRequest,
                {
                    "session_id": session_id,
                    "cwd": cwd,
                    "additional_directories": additional_directories,
                    "mcp_servers": mcp_servers,
                    "replay_from": replay_from,
                },
                kwargs,
            ),
        )

    @param_model(schema.CloseSessionRequest)
    async def close_session(self, session_id: str, **kwargs: Any) -> schema.CloseSessionResponse:
        return await self._request(
            AGENT_METHODS["session_close"],
            build_request(schema.CloseSessionRequest, {"session_id": session_id}, kwargs),
        )

    @param_model(SetConfigOptionRequest)
    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: Any,
        *,
        type: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> schema.SetSessionConfigOptionResponse:
        request = build_config_request(config_id, session_id, value, type, kwargs)
        return await self._request(AGENT_METHODS["session_set_config_option"], request)

    @param_model(schema.PromptRequest)
    async def prompt(
        self,
        session_id: str,
        prompt: list[
            schema.TextContentBlock
            | schema.ImageContentBlock
            | schema.AudioContentBlock
            | schema.ResourceContentBlock
            | schema.EmbeddedResourceContentBlock
            | schema.OtherContentBlock
        ],
        **kwargs: Any,
    ) -> schema.PromptResponse:
        return await self._request(
            AGENT_METHODS["session_prompt"],
            build_request(schema.PromptRequest, {"session_id": session_id, "prompt": prompt}, kwargs),
        )

    @param_model(schema.CancelSessionNotification)
    async def cancel_session(self, session_id: str, **kwargs: Any) -> None:
        await self._notify(
            AGENT_METHODS["session_cancel"],
            build_request(schema.CancelSessionNotification, {"session_id": session_id}, kwargs),
        )

    @param_model(schema.MessageMcpRequest)
    async def mcp_message(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        return await self._request(
            AGENT_METHODS["mcp_message"],
            build_request(
                schema.MessageMcpRequest, {"connection_id": connection_id, "method": method, "params": params}, kwargs
            ),
        )

    @param_model(schema.MessageMcpNotification)
    async def notify_mcp(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        await self._notify(
            AGENT_METHODS["mcp_message"],
            build_request(
                schema.MessageMcpNotification,
                {"connection_id": connection_id, "method": method, "params": params},
                kwargs,
            ),
        )

    @param_model(schema.StartNesRequest)
    async def start_nes(
        self,
        workspace_uri: str | AnyUrl | None = None,
        workspace_folders: list[schema.WorkspaceFolder] | None = None,
        repository: schema.NesRepository | None = None,
        **kwargs: Any,
    ) -> schema.StartNesResponse:
        return await self._request(
            AGENT_METHODS["nes_start"],
            build_request(
                schema.StartNesRequest,
                {"workspace_uri": workspace_uri, "workspace_folders": workspace_folders, "repository": repository},
                kwargs,
            ),
        )

    @param_model(schema.SuggestNesRequest)
    async def suggest_nes(
        self,
        session_id: str,
        uri: str | AnyUrl,
        version: int,
        position: schema.Position,
        trigger_kind: Literal["automatic"] | Literal["diagnostic"] | Literal["manual"] | str,
        selection: schema.Range | None = None,
        context: schema.NesSuggestContext | None = None,
        **kwargs: Any,
    ) -> schema.SuggestNesResponse:
        return await self._request(
            AGENT_METHODS["nes_suggest"],
            build_request(
                schema.SuggestNesRequest,
                {
                    "session_id": session_id,
                    "uri": uri,
                    "version": version,
                    "position": position,
                    "trigger_kind": trigger_kind,
                    "selection": selection,
                    "context": context,
                },
                kwargs,
            ),
        )

    @param_model(schema.AcceptNesNotification)
    async def accept_nes(self, session_id: str, suggestion_id: str, **kwargs: Any) -> None:
        await self._notify(
            AGENT_METHODS["nes_accept"],
            build_request(
                schema.AcceptNesNotification, {"session_id": session_id, "suggestion_id": suggestion_id}, kwargs
            ),
        )

    @param_model(schema.RejectNesNotification)
    async def reject_nes(
        self,
        session_id: str,
        suggestion_id: str,
        reason: Literal["rejected"]
        | Literal["ignored"]
        | Literal["replaced"]
        | Literal["cancelled"]
        | str
        | None = None,
        **kwargs: Any,
    ) -> None:
        await self._notify(
            AGENT_METHODS["nes_reject"],
            build_request(
                schema.RejectNesNotification,
                {"session_id": session_id, "suggestion_id": suggestion_id, "reason": reason},
                kwargs,
            ),
        )

    @param_model(schema.CloseNesRequest)
    async def close_nes(self, session_id: str, **kwargs: Any) -> schema.CloseNesResponse:
        return await self._request(
            AGENT_METHODS["nes_close"], build_request(schema.CloseNesRequest, {"session_id": session_id}, kwargs)
        )

    @param_model(schema.DidOpenDocumentNotification)
    async def did_open(
        self, session_id: str, uri: str | AnyUrl, language_id: str, version: int, text: str, **kwargs: Any
    ) -> None:
        await self._notify(
            AGENT_METHODS["document_did_open"],
            build_request(
                schema.DidOpenDocumentNotification,
                {"session_id": session_id, "uri": uri, "language_id": language_id, "version": version, "text": text},
                kwargs,
            ),
        )

    @param_model(schema.DidChangeDocumentNotification)
    async def did_change(
        self,
        session_id: str,
        uri: str | AnyUrl,
        version: int,
        content_changes: list[schema.TextDocumentContentChangeEvent],
        **kwargs: Any,
    ) -> None:
        await self._notify(
            AGENT_METHODS["document_did_change"],
            build_request(
                schema.DidChangeDocumentNotification,
                {"session_id": session_id, "uri": uri, "version": version, "content_changes": content_changes},
                kwargs,
            ),
        )

    @param_model(schema.DidCloseDocumentNotification)
    async def did_close(self, session_id: str, uri: str | AnyUrl, **kwargs: Any) -> None:
        await self._notify(
            AGENT_METHODS["document_did_close"],
            build_request(schema.DidCloseDocumentNotification, {"session_id": session_id, "uri": uri}, kwargs),
        )

    @param_model(schema.DidSaveDocumentNotification)
    async def did_save(self, session_id: str, uri: str | AnyUrl, **kwargs: Any) -> None:
        await self._notify(
            AGENT_METHODS["document_did_save"],
            build_request(schema.DidSaveDocumentNotification, {"session_id": session_id, "uri": uri}, kwargs),
        )

    @param_model(schema.DidFocusDocumentNotification)
    async def did_focus(
        self,
        session_id: str,
        uri: str | AnyUrl,
        version: int,
        position: schema.Position,
        visible_range: schema.Range,
        **kwargs: Any,
    ) -> None:
        await self._notify(
            AGENT_METHODS["document_did_focus"],
            build_request(
                schema.DidFocusDocumentNotification,
                {
                    "session_id": session_id,
                    "uri": uri,
                    "version": version,
                    "position": position,
                    "visible_range": visible_range,
                },
                kwargs,
            ),
        )

    async def send_extension_request(self, method: str, params: Any = None) -> Any:
        await self._state.require(method)
        return await self._conn.send_request(_extension_method(method), params)

    async def send_extension_notification(self, method: str, params: Any = None) -> None:
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
    client: object,
    input_stream: Any,
    output_stream: Any = None,
    **connection_kwargs: Any,
) -> ClientSideConnection:
    return ClientSideConnection(client, input_stream, output_stream, **connection_kwargs)
