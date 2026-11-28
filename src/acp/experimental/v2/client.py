from __future__ import annotations

from typing import Any, Literal

from pydantic import AnyUrl, BaseModel

from acp.utils import param_model

from . import schema
from ._connection import open_connection
from ._initialization import InitializationState
from ._methods import AGENT_REQUESTS_BY_METHOD
from ._params import build_config_request, build_request
from ._router import MethodRouter
from ._schema_base import WIRE_VALIDATION_CONTEXT
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
            parsed = schema.InitializeResponse.model_validate(response, context=WIRE_VALIDATION_CONTEXT)
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
        system_prompt: schema.SystemPrompt | None = None,
        config_options: list[
            schema.InitialIdConfigOption | schema.InitialBooleanConfigOption | schema.OtherInitialConfigOption
        ]
        | None = None,
        **kwargs: Any,
    ) -> schema.NewSessionResponse:
        return await self._request(
            AGENT_METHODS["session_new"],
            build_request(
                schema.NewSessionRequest,
                {
                    "cwd": cwd,
                    "additional_directories": additional_directories,
                    "mcp_servers": mcp_servers,
                    "system_prompt": system_prompt,
                    "config_options": config_options,
                },
                kwargs,
            ),
        )

    @param_model(schema.ListSessionsRequest)
    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        archived: Literal["exclude"] | Literal["include"] | Literal["only"] | str | None = None,
        **kwargs: Any,
    ) -> schema.ListSessionsResponse:
        return await self._request(
            AGENT_METHODS["session_list"],
            build_request(schema.ListSessionsRequest, {"cwd": cwd, "cursor": cursor, "archived": archived}, kwargs),
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
        system_prompt: schema.SystemPrompt | None = None,
        config_options: list[
            schema.InitialIdConfigOption | schema.InitialBooleanConfigOption | schema.OtherInitialConfigOption
        ]
        | None = None,
        cursor_id: str | None = None,
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
                    "system_prompt": system_prompt,
                    "config_options": config_options,
                    "cursor_id": cursor_id,
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
        system_prompt: schema.SystemPrompt | None = None,
        config_options: list[
            schema.InitialIdConfigOption | schema.InitialBooleanConfigOption | schema.OtherInitialConfigOption
        ]
        | None = None,
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
                    "system_prompt": system_prompt,
                    "config_options": config_options,
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
        self, session_id: str, prompt: list[schema.ContentBlock], idempotency_key: str | None = None, **kwargs: Any
    ) -> schema.PromptResponse:
        return await self._request(
            AGENT_METHODS["session_prompt"],
            build_request(
                schema.PromptRequest,
                {"session_id": session_id, "prompt": prompt, "idempotency_key": idempotency_key},
                kwargs,
            ),
        )

    @param_model(schema.CancelSessionNotification)
    async def cancel_session(self, session_id: str, **kwargs: Any) -> None:
        await self._notify(
            AGENT_METHODS["session_cancel"],
            build_request(schema.CancelSessionNotification, {"session_id": session_id}, kwargs),
        )

    @param_model(schema.MessageMcpNotification)
    async def notify_mcp(
        self, server_id: str, request_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        await self._notify(
            AGENT_METHODS["mcp_message"],
            build_request(
                schema.MessageMcpNotification,
                {"server_id": server_id, "request_id": request_id, "method": method, "params": params},
                kwargs,
            ),
        )

    @param_model(schema.AuthStatusRequest)
    async def auth_status(self, **kwargs: Any) -> schema.AuthStatusResponse:
        return await self._request(AGENT_METHODS["auth_status"], build_request(schema.AuthStatusRequest, {}, kwargs))

    @param_model(schema.ArchiveSessionRequest)
    async def archive_session(self, session_id: str, **kwargs: Any) -> schema.ArchiveSessionResponse:
        return await self._request(
            AGENT_METHODS["session_archive"],
            build_request(schema.ArchiveSessionRequest, {"session_id": session_id}, kwargs),
        )

    @param_model(schema.UnarchiveSessionRequest)
    async def unarchive_session(self, session_id: str, **kwargs: Any) -> schema.UnarchiveSessionResponse:
        return await self._request(
            AGENT_METHODS["session_unarchive"],
            build_request(schema.UnarchiveSessionRequest, {"session_id": session_id}, kwargs),
        )

    @param_model(schema.SessionStatusRequest)
    async def session_status(self, session_id: str, **kwargs: Any) -> schema.SessionStatusResponse:
        return await self._request(
            AGENT_METHODS["session_status"],
            build_request(schema.SessionStatusRequest, {"session_id": session_id}, kwargs),
        )

    @param_model(schema.SessionOptionsRequest)
    async def session_options(
        self, cwd: str, additional_directories: list[str] | None = None, session_id: str | None = None, **kwargs: Any
    ) -> schema.SessionOptionsResponse:
        return await self._request(
            AGENT_METHODS["session_options"],
            build_request(
                schema.SessionOptionsRequest,
                {"cwd": cwd, "additional_directories": additional_directories, "session_id": session_id},
                kwargs,
            ),
        )

    @param_model(schema.SetSessionTitleRequest)
    async def set_session_title(self, session_id: str, title: str, **kwargs: Any) -> schema.SetSessionTitleResponse:
        return await self._request(
            AGENT_METHODS["session_set_title"],
            build_request(schema.SetSessionTitleRequest, {"session_id": session_id, "title": title}, kwargs),
        )

    @param_model(schema.CancelToolCallRequest)
    async def cancel_tool_call(
        self, session_id: str, tool_call_id: str, **kwargs: Any
    ) -> schema.CancelToolCallResponse:
        return await self._request(
            AGENT_METHODS["tool_call_cancel"],
            build_request(
                schema.CancelToolCallRequest, {"session_id": session_id, "tool_call_id": tool_call_id}, kwargs
            ),
        )

    @param_model(schema.CancelTerminalRequest)
    async def cancel_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> schema.CancelTerminalResponse:
        return await self._request(
            AGENT_METHODS["terminal_cancel"],
            build_request(schema.CancelTerminalRequest, {"session_id": session_id, "terminal_id": terminal_id}, kwargs),
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
        return spec.response.validate_python(response, context=WIRE_VALIDATION_CONTEXT)

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
