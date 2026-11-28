from __future__ import annotations

from functools import partial
from typing import Any, Literal, Protocol

from pydantic import AnyUrl, TypeAdapter

from acp._mcp import validate_mcp_response
from acp.utils import param_model

from . import schema
from .meta import AGENT_METHODS, CLIENT_METHODS

SetConfigOptionRequest = schema.SetSessionConfigOptionRequest
CreateElicitationRequest = schema.CreateElicitationRequest
CreateElicitationResponse = schema.CreateElicitationResponse


class Agent(Protocol):
    """Agent handlers for ACP v2; only override the methods you support."""

    @param_model(schema.InitializeRequest, method=AGENT_METHODS["initialize"])
    async def initialize(
        self,
        protocol_version: int,
        info: schema.Implementation,
        capabilities: schema.ClientCapabilities | None = None,
        **kwargs: Any,
    ) -> schema.InitializeResponse: ...

    @param_model(schema.LoginAuthRequest, method=AGENT_METHODS["auth_login"], default_result={})
    async def login(self, method_id: str, **kwargs: Any) -> schema.LoginAuthResponse: ...

    @param_model(schema.LogoutAuthRequest, method=AGENT_METHODS["auth_logout"], default_result={})
    async def logout(self, **kwargs: Any) -> schema.LogoutAuthResponse: ...

    @param_model(schema.ListProvidersRequest, method=AGENT_METHODS["providers_list"])
    async def list_providers(self, **kwargs: Any) -> schema.ListProvidersResponse: ...

    @param_model(schema.SetProviderRequest, method=AGENT_METHODS["providers_set"], default_result={})
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
    ) -> schema.SetProviderResponse: ...

    @param_model(schema.DisableProviderRequest, method=AGENT_METHODS["providers_disable"], default_result={})
    async def disable_provider(self, provider_id: str, **kwargs: Any) -> schema.DisableProviderResponse: ...

    @param_model(schema.NewSessionRequest, method=AGENT_METHODS["session_new"])
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
    ) -> schema.NewSessionResponse: ...

    @param_model(schema.ListSessionsRequest, method=AGENT_METHODS["session_list"])
    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        archived: Literal["exclude"] | Literal["include"] | Literal["only"] | str | None = None,
        **kwargs: Any,
    ) -> schema.ListSessionsResponse: ...

    @param_model(schema.DeleteSessionRequest, method=AGENT_METHODS["session_delete"], default_result={})
    async def delete_session(self, session_id: str, **kwargs: Any) -> schema.DeleteSessionResponse: ...

    @param_model(schema.ForkSessionRequest, method=AGENT_METHODS["session_fork"])
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
    ) -> schema.ForkSessionResponse: ...

    @param_model(schema.ResumeSessionRequest, method=AGENT_METHODS["session_resume"])
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
    ) -> schema.ResumeSessionResponse: ...

    @param_model(schema.CloseSessionRequest, method=AGENT_METHODS["session_close"], default_result={})
    async def close_session(self, session_id: str, **kwargs: Any) -> schema.CloseSessionResponse: ...

    @param_model(SetConfigOptionRequest, method=AGENT_METHODS["session_set_config_option"])
    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: Any,
        *,
        type: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> schema.SetSessionConfigOptionResponse: ...

    @param_model(schema.PromptRequest, method=AGENT_METHODS["session_prompt"])
    async def prompt(
        self, session_id: str, prompt: list[schema.ContentBlock], idempotency_key: str | None = None, **kwargs: Any
    ) -> schema.PromptResponse: ...

    @param_model(schema.CancelSessionNotification, method=AGENT_METHODS["session_cancel"], kind="notification")
    async def cancel_session(self, session_id: str, **kwargs: Any) -> None: ...

    @param_model(schema.MessageMcpNotification, method=AGENT_METHODS["mcp_message"], kind="notification")
    async def notify_mcp(
        self, server_id: str, request_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None: ...

    @param_model(schema.AuthStatusRequest, method=AGENT_METHODS["auth_status"])
    async def auth_status(self, **kwargs: Any) -> schema.AuthStatusResponse: ...

    @param_model(schema.ArchiveSessionRequest, method=AGENT_METHODS["session_archive"], default_result={})
    async def archive_session(self, session_id: str, **kwargs: Any) -> schema.ArchiveSessionResponse: ...

    @param_model(schema.UnarchiveSessionRequest, method=AGENT_METHODS["session_unarchive"], default_result={})
    async def unarchive_session(self, session_id: str, **kwargs: Any) -> schema.UnarchiveSessionResponse: ...

    @param_model(schema.SessionStatusRequest, method=AGENT_METHODS["session_status"])
    async def session_status(self, session_id: str, **kwargs: Any) -> schema.SessionStatusResponse: ...

    @param_model(schema.SessionOptionsRequest, method=AGENT_METHODS["session_options"])
    async def session_options(
        self, cwd: str, additional_directories: list[str] | None = None, session_id: str | None = None, **kwargs: Any
    ) -> schema.SessionOptionsResponse: ...

    @param_model(schema.SetSessionTitleRequest, method=AGENT_METHODS["session_set_title"])
    async def set_session_title(self, session_id: str, title: str, **kwargs: Any) -> schema.SetSessionTitleResponse: ...

    @param_model(schema.CancelToolCallRequest, method=AGENT_METHODS["tool_call_cancel"], default_result={})
    async def cancel_tool_call(
        self, session_id: str, tool_call_id: str, **kwargs: Any
    ) -> schema.CancelToolCallResponse: ...

    @param_model(schema.CancelTerminalRequest, method=AGENT_METHODS["terminal_cancel"], default_result={})
    async def cancel_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> schema.CancelTerminalResponse: ...


class Client(Protocol):
    """Client handlers for ACP v2; only override the methods you support."""

    @param_model(schema.RequestPermissionRequest, method=CLIENT_METHODS["session_request_permission"])
    async def request_permission(
        self,
        session_id: str,
        title: str,
        options: list[schema.PermissionOption],
        description: str | None = None,
        subject: schema.ToolCallPermissionSubjectVariant
        | schema.CommandPermissionSubjectVariant
        | schema.OtherPermissionSubject
        | None = None,
        **kwargs: Any,
    ) -> schema.RequestPermissionResponse: ...

    @param_model(schema.UpdateSessionNotification, method=CLIENT_METHODS["session_update"], kind="notification")
    async def session_update(self, session_id: str, update: schema.SessionUpdate, **kwargs: Any) -> None: ...

    @param_model(
        schema.MessageMcpRequest,
        method=CLIENT_METHODS["mcp_message"],
        adapt_result=partial(validate_mcp_response, adapter=TypeAdapter(schema.MessageMcpResponse)),
    )
    async def mcp_message(
        self,
        server_id: str,
        request_id: str,
        method: str,
        params: dict[str, Any] | None = None,
        session_id: str | None = None,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> schema.MessageMcpResponse: ...

    @param_model(CreateElicitationRequest, method=CLIENT_METHODS["elicitation_create"])
    async def create_elicitation(
        self,
        message: str,
        mode: str,
        *,
        session_id: str | None = None,
        request_id: int | str | None = None,
        tool_call_id: str | None = None,
        requested_schema: schema.ElicitationSchema | None = None,
        elicitation_id: str | None = None,
        url: str | AnyUrl | None = None,
        **kwargs: Any,
    ) -> CreateElicitationResponse: ...

    @param_model(
        schema.CompleteElicitationNotification, method=CLIENT_METHODS["elicitation_complete"], kind="notification"
    )
    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None: ...

    @param_model(schema.AuthStatusUpdateNotification, method=CLIENT_METHODS["auth_status_update"], kind="notification")
    async def auth_status_update(
        self,
        state: Literal["authenticated"]
        | Literal["unauthenticated"]
        | Literal["not_required"]
        | Literal["unknown"]
        | str,
        accounts: list[schema.AuthAccount] | None = None,
        **kwargs: Any,
    ) -> None: ...
