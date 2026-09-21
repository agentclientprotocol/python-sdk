from __future__ import annotations

from typing import Any, Literal, Protocol

from .schema import (
    AcceptNesNotification,
    AcpMcpServer,
    AgentMessageChunk,
    AgentPlanContentUpdate,
    AgentPlanRemovedUpdate,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AudioContentBlock,
    AuthenticateRequest,
    AuthenticateResponse,
    AvailableCommandsUpdate,
    CancelNotification,
    ClientCapabilities,
    CloseNesRequest,
    CloseNesResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    CompleteElicitationNotification,
    ConfigOptionUpdate,
    ConnectMcpRequest,
    ConnectMcpResponse,
    CreateElicitationResponse,
    CreateTerminalRequest,
    CreateTerminalResponse,
    CurrentModeUpdate,
    DeleteSessionRequest,
    DeleteSessionResponse,
    DidChangeDocumentNotification,
    DidCloseDocumentNotification,
    DidFocusDocumentNotification,
    DidOpenDocumentNotification,
    DidSaveDocumentNotification,
    DisableProviderRequest,
    DisableProviderResponse,
    DisconnectMcpRequest,
    DisconnectMcpResponse,
    ElicitationMode,
    EmbeddedResourceContentBlock,
    EnvVariable,
    ForkSessionRequest,
    ForkSessionResponse,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    InitializeRequest,
    InitializeResponse,
    KillTerminalRequest,
    KillTerminalResponse,
    ListProvidersRequest,
    ListProvidersResponse,
    ListSessionsRequest,
    ListSessionsResponse,
    LoadSessionRequest,
    LoadSessionResponse,
    LogoutRequest,
    LogoutResponse,
    McpServerStdio,
    MessageMcpNotification,
    MessageMcpRequest,
    NesRepository,
    NesSuggestContext,
    NewSessionRequest,
    NewSessionResponse,
    PermissionOption,
    Position,
    PromptRequest,
    PromptResponse,
    Range,
    ReadTextFileRequest,
    ReadTextFileResponse,
    RejectNesNotification,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
    ResourceContentBlock,
    ResumeSessionRequest,
    ResumeSessionResponse,
    SessionInfoUpdate,
    SessionNotification,
    SessionUpdateCompactionSummaryChunk,
    SessionUpdateCompactionUpdate,
    SessionUpdateNotice,
    SetProviderRequest,
    SetProviderResponse,
    SetSessionConfigOptionBooleanRequest,
    SetSessionConfigOptionResponse,
    SetSessionConfigOptionSelectRequest,
    SetSessionModeRequest,
    SetSessionModeResponse,
    SseMcpServer,
    StartNesRequest,
    StartNesResponse,
    SuggestNesRequest,
    SuggestNesResponse,
    TerminalOutputRequest,
    TerminalOutputResponse,
    TextContentBlock,
    TextDocumentContentChangeEvent,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UsageUpdate,
    UserMessageChunk,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
    WorkspaceFolder,
    WriteTextFileRequest,
    WriteTextFileResponse,
)
from .utils import param_model, param_models

__all__ = ["Agent", "Client"]


class Client(Protocol):
    @param_model(ConnectMcpRequest)
    async def connect_mcp(self, server_id: str, **kwargs: Any) -> ConnectMcpResponse: ...

    @param_model(DisconnectMcpRequest)
    async def disconnect_mcp(self, connection_id: str, **kwargs: Any) -> DisconnectMcpResponse: ...

    @param_model(MessageMcpRequest)
    async def mcp_message(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any: ...

    @param_model(MessageMcpNotification)
    async def notify_mcp(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None: ...

    @param_model(RequestPermissionRequest)
    async def request_permission(
        self, session_id: str, tool_call: ToolCallUpdate, options: list[PermissionOption], **kwargs: Any
    ) -> RequestPermissionResponse: ...

    @param_model(SessionNotification)
    async def session_update(
        self,
        session_id: str,
        update: UserMessageChunk
        | AgentMessageChunk
        | AgentThoughtChunk
        | ToolCallStart
        | ToolCallProgress
        | AgentPlanUpdate
        | AgentPlanContentUpdate
        | AgentPlanRemovedUpdate
        | AvailableCommandsUpdate
        | CurrentModeUpdate
        | ConfigOptionUpdate
        | SessionInfoUpdate
        | UsageUpdate
        | SessionUpdateNotice
        | SessionUpdateCompactionUpdate
        | SessionUpdateCompactionSummaryChunk,
        **kwargs: Any,
    ) -> None: ...

    @param_model(WriteTextFileRequest)
    async def write_text_file(
        self, session_id: str, path: str, content: str, **kwargs: Any
    ) -> WriteTextFileResponse | None: ...

    @param_model(ReadTextFileRequest)
    async def read_text_file(
        self, session_id: str, path: str, line: int | None = None, limit: int | None = None, **kwargs: Any
    ) -> ReadTextFileResponse: ...

    @param_model(CreateTerminalRequest)
    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: list[str] | None = None,
        env: list[EnvVariable] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse: ...

    @param_model(TerminalOutputRequest)
    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse: ...

    @param_model(ReleaseTerminalRequest)
    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None: ...

    @param_model(WaitForTerminalExitRequest)
    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse: ...

    @param_model(KillTerminalRequest)
    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> KillTerminalResponse | None: ...

    async def create_elicitation(
        self, message: str, mode: ElicitationMode, **kwargs: Any
    ) -> CreateElicitationResponse: ...

    @param_model(CompleteElicitationNotification)
    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None: ...

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None: ...

    def on_connect(self, conn: Agent) -> None: ...


class Agent(Protocol):
    @param_model(DeleteSessionRequest)
    async def delete_session(self, session_id: str, **kwargs: Any) -> DeleteSessionResponse: ...

    @param_model(ListProvidersRequest)
    async def list_providers(self, **kwargs: Any) -> ListProvidersResponse: ...

    @param_model(SetProviderRequest)
    async def set_provider(
        self,
        provider_id: str,
        api_type: Literal["anthropic"]
        | Literal["openai"]
        | Literal["azure"]
        | Literal["vertex"]
        | Literal["bedrock"]
        | str,
        base_url: str,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> SetProviderResponse: ...

    @param_model(DisableProviderRequest)
    async def disable_provider(self, provider_id: str, **kwargs: Any) -> DisableProviderResponse: ...

    @param_model(LogoutRequest)
    async def logout(self, **kwargs: Any) -> LogoutResponse: ...

    @param_model(MessageMcpRequest)
    async def mcp_message(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any: ...

    @param_model(MessageMcpNotification)
    async def notify_mcp(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None: ...

    @param_model(StartNesRequest)
    async def start_nes(
        self,
        workspace_uri: str | None = None,
        workspace_folders: list[WorkspaceFolder] | None = None,
        repository: NesRepository | None = None,
        **kwargs: Any,
    ) -> StartNesResponse: ...

    @param_model(SuggestNesRequest)
    async def suggest_nes(
        self,
        session_id: str,
        uri: str,
        version: int,
        position: Position,
        trigger_kind: Literal["automatic", "diagnostic", "manual"],
        selection: Range | None = None,
        context: NesSuggestContext | None = None,
        **kwargs: Any,
    ) -> SuggestNesResponse: ...

    @param_model(CloseNesRequest)
    async def close_nes(self, session_id: str, **kwargs: Any) -> CloseNesResponse: ...

    @param_model(AcceptNesNotification)
    async def accept_nes(self, session_id: str, id: str, **kwargs: Any) -> None: ...  # noqa: A002

    @param_model(RejectNesNotification)
    async def reject_nes(
        self,
        session_id: str,
        id: str,  # noqa: A002
        reason: Literal["rejected", "ignored", "replaced", "cancelled"] | None = None,
        **kwargs: Any,
    ) -> None: ...

    @param_model(DidOpenDocumentNotification)
    async def did_open(
        self, session_id: str, uri: str, language_id: str, version: int, text: str, **kwargs: Any
    ) -> None: ...

    @param_model(DidChangeDocumentNotification)
    async def did_change(
        self,
        session_id: str,
        uri: str,
        version: int,
        content_changes: list[TextDocumentContentChangeEvent],
        **kwargs: Any,
    ) -> None: ...

    @param_model(DidCloseDocumentNotification)
    async def did_close(self, session_id: str, uri: str, **kwargs: Any) -> None: ...

    @param_model(DidSaveDocumentNotification)
    async def did_save(self, session_id: str, uri: str, **kwargs: Any) -> None: ...

    @param_model(DidFocusDocumentNotification)
    async def did_focus(
        self, session_id: str, uri: str, version: int, position: Position, visible_range: Range, **kwargs: Any
    ) -> None: ...

    @param_model(InitializeRequest)
    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse: ...

    @param_model(NewSessionRequest)
    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse: ...

    @param_model(LoadSessionRequest)
    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None: ...

    @param_model(ListSessionsRequest)
    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse: ...

    @param_model(SetSessionModeRequest)
    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> SetSessionModeResponse | None: ...

    @param_models(SetSessionConfigOptionBooleanRequest, SetSessionConfigOptionSelectRequest)
    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None: ...

    @param_model(AuthenticateRequest)
    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None: ...

    @param_model(PromptRequest)
    async def prompt(
        self,
        session_id: str,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        **kwargs: Any,
    ) -> PromptResponse: ...

    @param_model(ForkSessionRequest)
    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ForkSessionResponse: ...

    @param_model(ResumeSessionRequest)
    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse: ...

    @param_model(CloseSessionRequest)
    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse | None: ...

    @param_model(CancelNotification)
    async def cancel(self, session_id: str, **kwargs: Any) -> None: ...

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None: ...

    def on_connect(self, conn: Client) -> None: ...
