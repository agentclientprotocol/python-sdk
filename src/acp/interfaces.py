from __future__ import annotations

from typing import Any, Literal, Protocol

from ._protocol_adapters import (
    elicitation_to_kwargs,
    validate_create_elicitation_request,
    validate_set_config_option_request,
)
from .meta import AGENT_METHODS, CLIENT_METHODS
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
    CreateFormRequestElicitationRequest,
    CreateFormSessionElicitationRequest,
    CreateTerminalRequest,
    CreateTerminalResponse,
    CreateUrlRequestElicitationRequest,
    CreateUrlSessionElicitationRequest,
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
from .utils import normalize_result, param_model

__all__ = ["Agent", "Client"]


class Client(Protocol):
    @param_model(ConnectMcpRequest, method=CLIENT_METHODS["mcp_connect"], unstable=True)
    async def connect_mcp(self, server_id: str, **kwargs: Any) -> ConnectMcpResponse: ...

    @param_model(
        DisconnectMcpRequest, method=CLIENT_METHODS["mcp_disconnect"], unstable=True, adapt_result=normalize_result
    )
    async def disconnect_mcp(self, connection_id: str, **kwargs: Any) -> DisconnectMcpResponse: ...

    @param_model(MessageMcpRequest, method=CLIENT_METHODS["mcp_message"], unstable=True)
    async def mcp_message(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any: ...

    @param_model(MessageMcpNotification, method=CLIENT_METHODS["mcp_message"], kind="notification", unstable=True)
    async def notify_mcp(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None: ...

    @param_model(RequestPermissionRequest, method=CLIENT_METHODS["session_request_permission"])
    async def request_permission(
        self, session_id: str, tool_call: ToolCallUpdate, options: list[PermissionOption], **kwargs: Any
    ) -> RequestPermissionResponse: ...

    @param_model(SessionNotification, method=CLIENT_METHODS["session_update"], kind="notification")
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

    @param_model(WriteTextFileRequest, method=CLIENT_METHODS["fs_write_text_file"])
    async def write_text_file(
        self, session_id: str, path: str, content: str, **kwargs: Any
    ) -> WriteTextFileResponse | None: ...

    @param_model(ReadTextFileRequest, method=CLIENT_METHODS["fs_read_text_file"])
    async def read_text_file(
        self, session_id: str, path: str, line: int | None = None, limit: int | None = None, **kwargs: Any
    ) -> ReadTextFileResponse: ...

    @param_model(CreateTerminalRequest, method=CLIENT_METHODS["terminal_create"], optional=True, default_result=None)
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

    @param_model(TerminalOutputRequest, method=CLIENT_METHODS["terminal_output"], optional=True, default_result=None)
    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse: ...

    @param_model(
        ReleaseTerminalRequest,
        method=CLIENT_METHODS["terminal_release"],
        optional=True,
        default_result={},
        adapt_result=normalize_result,
    )
    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None: ...

    @param_model(
        WaitForTerminalExitRequest, method=CLIENT_METHODS["terminal_wait_for_exit"], optional=True, default_result=None
    )
    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse: ...

    @param_model(
        KillTerminalRequest,
        method=CLIENT_METHODS["terminal_kill"],
        optional=True,
        default_result={},
        adapt_result=normalize_result,
    )
    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> KillTerminalResponse | None: ...

    @param_model(
        CreateFormSessionElicitationRequest
        | CreateFormRequestElicitationRequest
        | CreateUrlSessionElicitationRequest
        | CreateUrlRequestElicitationRequest,
        method=CLIENT_METHODS["elicitation_create"],
        unstable=True,
        validate_params=validate_create_elicitation_request,
        adapt_params=elicitation_to_kwargs,
        adapt_result=normalize_result,
    )
    async def create_elicitation(
        self, message: str, mode: ElicitationMode, **kwargs: Any
    ) -> CreateElicitationResponse: ...

    @param_model(
        CompleteElicitationNotification,
        method=CLIENT_METHODS["elicitation_complete"],
        kind="notification",
        unstable=True,
    )
    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None: ...

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None: ...

    def on_connect(self, conn: Agent) -> None: ...


class Agent(Protocol):
    @param_model(DeleteSessionRequest, method=AGENT_METHODS["session_delete"], adapt_result=normalize_result)
    async def delete_session(self, session_id: str, **kwargs: Any) -> DeleteSessionResponse: ...

    @param_model(ListProvidersRequest, method=AGENT_METHODS["providers_list"], unstable=True)
    async def list_providers(self, **kwargs: Any) -> ListProvidersResponse: ...

    @param_model(
        SetProviderRequest, method=AGENT_METHODS["providers_set"], unstable=True, adapt_result=normalize_result
    )
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

    @param_model(
        DisableProviderRequest, method=AGENT_METHODS["providers_disable"], unstable=True, adapt_result=normalize_result
    )
    async def disable_provider(self, provider_id: str, **kwargs: Any) -> DisableProviderResponse: ...

    @param_model(LogoutRequest, method=AGENT_METHODS["logout"], adapt_result=normalize_result)
    async def logout(self, **kwargs: Any) -> LogoutResponse: ...

    @param_model(MessageMcpRequest, method=AGENT_METHODS["mcp_message"], unstable=True)
    async def mcp_message(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any: ...

    @param_model(MessageMcpNotification, method=AGENT_METHODS["mcp_message"], kind="notification", unstable=True)
    async def notify_mcp(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None: ...

    @param_model(StartNesRequest, method=AGENT_METHODS["nes_start"], unstable=True)
    async def start_nes(
        self,
        workspace_uri: str | None = None,
        workspace_folders: list[WorkspaceFolder] | None = None,
        repository: NesRepository | None = None,
        **kwargs: Any,
    ) -> StartNesResponse: ...

    @param_model(SuggestNesRequest, method=AGENT_METHODS["nes_suggest"], unstable=True)
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

    @param_model(CloseNesRequest, method=AGENT_METHODS["nes_close"], unstable=True, adapt_result=normalize_result)
    async def close_nes(self, session_id: str, **kwargs: Any) -> CloseNesResponse: ...

    @param_model(AcceptNesNotification, method=AGENT_METHODS["nes_accept"], kind="notification", unstable=True)
    async def accept_nes(self, session_id: str, id: str, **kwargs: Any) -> None: ...  # noqa: A002

    @param_model(RejectNesNotification, method=AGENT_METHODS["nes_reject"], kind="notification", unstable=True)
    async def reject_nes(
        self,
        session_id: str,
        id: str,  # noqa: A002
        reason: Literal["rejected", "ignored", "replaced", "cancelled"] | None = None,
        **kwargs: Any,
    ) -> None: ...

    @param_model(
        DidOpenDocumentNotification, method=AGENT_METHODS["document_did_open"], kind="notification", unstable=True
    )
    async def did_open(
        self, session_id: str, uri: str, language_id: str, version: int, text: str, **kwargs: Any
    ) -> None: ...

    @param_model(
        DidChangeDocumentNotification, method=AGENT_METHODS["document_did_change"], kind="notification", unstable=True
    )
    async def did_change(
        self,
        session_id: str,
        uri: str,
        version: int,
        content_changes: list[TextDocumentContentChangeEvent],
        **kwargs: Any,
    ) -> None: ...

    @param_model(
        DidCloseDocumentNotification, method=AGENT_METHODS["document_did_close"], kind="notification", unstable=True
    )
    async def did_close(self, session_id: str, uri: str, **kwargs: Any) -> None: ...

    @param_model(
        DidSaveDocumentNotification, method=AGENT_METHODS["document_did_save"], kind="notification", unstable=True
    )
    async def did_save(self, session_id: str, uri: str, **kwargs: Any) -> None: ...

    @param_model(
        DidFocusDocumentNotification, method=AGENT_METHODS["document_did_focus"], kind="notification", unstable=True
    )
    async def did_focus(
        self, session_id: str, uri: str, version: int, position: Position, visible_range: Range, **kwargs: Any
    ) -> None: ...

    @param_model(InitializeRequest, method=AGENT_METHODS["initialize"])
    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse: ...

    @param_model(NewSessionRequest, method=AGENT_METHODS["session_new"])
    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse: ...

    @param_model(LoadSessionRequest, method=AGENT_METHODS["session_load"], adapt_result=normalize_result)
    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None: ...

    @param_model(ListSessionsRequest, method=AGENT_METHODS["session_list"])
    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse: ...

    @param_model(SetSessionModeRequest, method=AGENT_METHODS["session_set_mode"], adapt_result=normalize_result)
    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> SetSessionModeResponse | None: ...

    @param_model(
        SetSessionConfigOptionBooleanRequest | SetSessionConfigOptionSelectRequest,
        method=AGENT_METHODS["session_set_config_option"],
        validate_params=validate_set_config_option_request,
        adapt_result=normalize_result,
    )
    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None: ...

    @param_model(AuthenticateRequest, method=AGENT_METHODS["authenticate"], adapt_result=normalize_result)
    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None: ...

    @param_model(PromptRequest, method=AGENT_METHODS["session_prompt"])
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

    @param_model(ForkSessionRequest, method=AGENT_METHODS["session_fork"], unstable=True)
    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ForkSessionResponse: ...

    @param_model(ResumeSessionRequest, method=AGENT_METHODS["session_resume"], unstable=True)
    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse: ...

    @param_model(
        CloseSessionRequest, method=AGENT_METHODS["session_close"], unstable=True, adapt_result=normalize_result
    )
    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse | None: ...

    @param_model(CancelNotification, method=AGENT_METHODS["session_cancel"], kind="notification")
    async def cancel(self, session_id: str, **kwargs: Any) -> None: ...

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None: ...

    def on_connect(self, conn: Client) -> None: ...
