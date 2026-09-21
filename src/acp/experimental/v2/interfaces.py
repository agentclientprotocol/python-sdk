from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import AnyUrl

from acp.utils import param_model

from . import schema
from .meta import AGENT_METHODS, CLIENT_METHODS

SetConfigOptionRequest = (
    schema.SetSessionConfigOptionIdRequest
    | schema.SetSessionConfigOptionBooleanRequest
    | schema.SetSessionConfigOptionOtherRequest
)

CreateElicitationRequest = (
    schema.CreateOtherSessionElicitationRequest
    | schema.CreateOtherRequestElicitationRequest
    | schema.CreateFormSessionElicitationRequest
    | schema.CreateFormRequestElicitationRequest
    | schema.CreateUrlSessionElicitationRequest
    | schema.CreateUrlRequestElicitationRequest
)

CreateElicitationResponse = (
    schema.AcceptElicitationResponse
    | schema.DeclineElicitationResponse
    | schema.CancelElicitationResponse
    | schema.OtherElicitationResponse
)


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
        **kwargs: Any,
    ) -> schema.NewSessionResponse: ...

    @param_model(schema.ListSessionsRequest, method=AGENT_METHODS["session_list"])
    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
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
    ) -> schema.PromptResponse: ...

    @param_model(schema.CancelSessionNotification, method=AGENT_METHODS["session_cancel"], kind="notification")
    async def cancel_session(self, session_id: str, **kwargs: Any) -> None: ...

    @param_model(schema.MessageMcpRequest, method=AGENT_METHODS["mcp_message"])
    async def mcp_message(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any: ...

    @param_model(schema.MessageMcpNotification, method=AGENT_METHODS["mcp_message"], kind="notification")
    async def notify_mcp(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None: ...

    @param_model(schema.StartNesRequest, method=AGENT_METHODS["nes_start"])
    async def start_nes(
        self,
        workspace_uri: str | AnyUrl | None = None,
        workspace_folders: list[schema.WorkspaceFolder] | None = None,
        repository: schema.NesRepository | None = None,
        **kwargs: Any,
    ) -> schema.StartNesResponse: ...

    @param_model(schema.SuggestNesRequest, method=AGENT_METHODS["nes_suggest"])
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
    ) -> schema.SuggestNesResponse: ...

    @param_model(schema.AcceptNesNotification, method=AGENT_METHODS["nes_accept"], kind="notification")
    async def accept_nes(self, session_id: str, suggestion_id: str, **kwargs: Any) -> None: ...

    @param_model(schema.RejectNesNotification, method=AGENT_METHODS["nes_reject"], kind="notification")
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
    ) -> None: ...

    @param_model(schema.CloseNesRequest, method=AGENT_METHODS["nes_close"], default_result={})
    async def close_nes(self, session_id: str, **kwargs: Any) -> schema.CloseNesResponse: ...

    @param_model(schema.DidOpenDocumentNotification, method=AGENT_METHODS["document_did_open"], kind="notification")
    async def did_open(
        self, session_id: str, uri: str | AnyUrl, language_id: str, version: int, text: str, **kwargs: Any
    ) -> None: ...

    @param_model(schema.DidChangeDocumentNotification, method=AGENT_METHODS["document_did_change"], kind="notification")
    async def did_change(
        self,
        session_id: str,
        uri: str | AnyUrl,
        version: int,
        content_changes: list[schema.TextDocumentContentChangeEvent],
        **kwargs: Any,
    ) -> None: ...

    @param_model(schema.DidCloseDocumentNotification, method=AGENT_METHODS["document_did_close"], kind="notification")
    async def did_close(self, session_id: str, uri: str | AnyUrl, **kwargs: Any) -> None: ...

    @param_model(schema.DidSaveDocumentNotification, method=AGENT_METHODS["document_did_save"], kind="notification")
    async def did_save(self, session_id: str, uri: str | AnyUrl, **kwargs: Any) -> None: ...

    @param_model(schema.DidFocusDocumentNotification, method=AGENT_METHODS["document_did_focus"], kind="notification")
    async def did_focus(
        self,
        session_id: str,
        uri: str | AnyUrl,
        version: int,
        position: schema.Position,
        visible_range: schema.Range,
        **kwargs: Any,
    ) -> None: ...


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
    async def session_update(
        self,
        session_id: str,
        update: schema.UserMessageChunk
        | schema.UserMessageUpdate
        | schema.AgentMessageChunk
        | schema.AgentMessageUpdate
        | schema.AgentThoughtChunk
        | schema.AgentThoughtUpdate
        | schema.ToolCallContentChunkUpdate
        | schema.SessionToolCallUpdate
        | schema.SessionTerminalUpdate
        | schema.SessionTerminalOutputChunk
        | schema.SessionPlanUpdate
        | schema.SessionPlanRemovedUpdate
        | schema.AvailableCommandsUpdate
        | schema.ConfigOptionUpdate
        | schema.SessionInfoUpdate
        | schema.UsageUpdate
        | schema.SessionNotice
        | schema.SessionCompactionUpdate
        | schema.SessionCompactionSummaryChunk
        | schema.OtherSessionUpdate
        | schema.RunningSessionStateUpdate
        | schema.IdleSessionStateUpdate
        | schema.RequiresActionSessionStateUpdate
        | schema.OtherSessionStateUpdate,
        **kwargs: Any,
    ) -> None: ...

    @param_model(schema.ConnectMcpRequest, method=CLIENT_METHODS["mcp_connect"])
    async def connect_mcp(self, server_id: str, **kwargs: Any) -> schema.ConnectMcpResponse: ...

    @param_model(schema.MessageMcpRequest, method=CLIENT_METHODS["mcp_message"])
    async def mcp_message(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any: ...

    @param_model(schema.MessageMcpNotification, method=CLIENT_METHODS["mcp_message"], kind="notification")
    async def notify_mcp(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None: ...

    @param_model(schema.DisconnectMcpRequest, method=CLIENT_METHODS["mcp_disconnect"], default_result={})
    async def disconnect_mcp(self, connection_id: str, **kwargs: Any) -> schema.DisconnectMcpResponse: ...

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
