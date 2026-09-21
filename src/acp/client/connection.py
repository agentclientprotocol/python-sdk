from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Literal, cast, final

from .._transport import Transport
from ..connection import Connection
from ..exceptions import RequestError
from ..interfaces import Agent, Client
from ..meta import AGENT_METHODS, CLIENT_METHODS
from ..router import _resolve_handler, _warn_legacy_handler
from ..schema import (
    AcceptNesNotification,
    AcpMcpServer,
    AudioContentBlock,
    AuthenticateRequest,
    AuthenticateResponse,
    CancelNotification,
    ClientCapabilities,
    CloseNesRequest,
    CloseNesResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    DeleteSessionRequest,
    DeleteSessionResponse,
    DidChangeDocumentNotification,
    DidCloseDocumentNotification,
    DidFocusDocumentNotification,
    DidOpenDocumentNotification,
    DidSaveDocumentNotification,
    DisableProviderRequest,
    DisableProviderResponse,
    EmbeddedResourceContentBlock,
    ForkSessionRequest,
    ForkSessionResponse,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    InitializeRequest,
    InitializeResponse,
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
    Position,
    PromptRequest,
    PromptResponse,
    Range,
    RejectNesNotification,
    ResourceContentBlock,
    ResumeSessionRequest,
    ResumeSessionResponse,
    SessionNotification,
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
    TextContentBlock,
    TextDocumentContentChangeEvent,
    WorkspaceFolder,
)
from ..utils import (
    compatible_class,
    notify_model,
    param_model,
    request_model,
    request_model_from_dict,
    serialize_params,
)
from .router import build_client_router

__all__ = ["ClientSideConnection"]
_CLIENT_CONNECTION_ERROR = "ClientSideConnection requires asyncio StreamWriter/StreamReader"


class _SessionUpdateTracker:
    """Client proxy that tracks in-flight session updates."""

    def __init__(self, client: Client) -> None:
        self._client = client
        self._session_update, self._session_update_attr, self._legacy_session_update = _resolve_handler(
            client, "session_update"
        )
        self._pending: dict[str, set[asyncio.Future[None]]] = {}
        self._current_update: ContextVar[asyncio.Future[None] | None] = ContextVar(
            "acp_current_session_update", default=None
        )

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        completed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        pending = self._pending.setdefault(session_id, set())
        pending.add(completed)
        token = self._current_update.set(completed)

        try:
            if self._session_update is None:
                raise RequestError.method_not_found(CLIENT_METHODS["session_update"])
            if self._legacy_session_update:
                _warn_legacy_handler(self._client, self._session_update_attr)
                notification = SessionNotification(session_id=session_id, update=update, field_meta=kwargs or None)
                await self._session_update(notification)
            else:
                await self._session_update(session_id=session_id, update=update, **kwargs)
        finally:
            self._current_update.reset(token)
            if not completed.done():
                completed.set_result(None)
            pending.discard(completed)
            if not pending:
                self._pending.pop(session_id, None)

    async def wait(self, session_id: str) -> None:
        # Snapshot before yielding so updates received after the response are
        # not associated with this prompt.
        current = self._current_update.get()
        notifications = tuple(
            completed for completed in self._pending.get(session_id, set()) if completed is not current
        )
        if notifications:
            await asyncio.gather(*(asyncio.shield(completed) for completed in notifications))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


@final
@compatible_class
class ClientSideConnection:
    """Client-side connection wrapper that dispatches JSON-RPC messages to an Agent implementation.
    The client can use this connection to communicate with the Agent so it behaves like an Agent.
    """

    def __init__(
        self,
        to_client: Callable[[Agent], Client] | Client,
        input_stream: Any,
        output_stream: Any = None,
        *,
        use_unstable_protocol: bool = False,
        **connection_kwargs: Any,
    ) -> None:
        client = to_client(self) if callable(to_client) else to_client
        self._session_updates = _SessionUpdateTracker(cast(Client, client))
        handler = build_client_router(cast(Client, self._session_updates), use_unstable_protocol=use_unstable_protocol)

        if isinstance(input_stream, Transport):
            if output_stream is not None:
                raise TypeError(_CLIENT_CONNECTION_ERROR)
            self._conn = Connection(handler, input_stream, **connection_kwargs)
        else:
            if not isinstance(input_stream, asyncio.StreamWriter) or not isinstance(
                output_stream, asyncio.StreamReader
            ):
                raise TypeError(_CLIENT_CONNECTION_ERROR)
            self._conn = Connection(handler, input_stream, output_stream, **connection_kwargs)
        if on_connect := getattr(client, "on_connect", None):
            on_connect(self)

    @param_model(InitializeRequest)
    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return await request_model(
            self._conn,
            AGENT_METHODS["initialize"],
            InitializeRequest(
                protocol_version=protocol_version,
                client_capabilities=client_capabilities or ClientCapabilities(),
                client_info=client_info,
                field_meta=kwargs or None,
            ),
            InitializeResponse,
        )

    @param_model(NewSessionRequest)
    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        resolved_mcp_servers = mcp_servers or []
        return await request_model(
            self._conn,
            AGENT_METHODS["session_new"],
            NewSessionRequest(
                cwd=cwd,
                additional_directories=additional_directories,
                mcp_servers=resolved_mcp_servers,
                field_meta=kwargs or None,
            ),
            NewSessionResponse,
        )

    @param_model(LoadSessionRequest)
    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        resolved_mcp_servers = mcp_servers or []
        return await request_model_from_dict(
            self._conn,
            AGENT_METHODS["session_load"],
            LoadSessionRequest(
                cwd=cwd,
                additional_directories=additional_directories,
                mcp_servers=resolved_mcp_servers,
                session_id=session_id,
                field_meta=kwargs or None,
            ),
            LoadSessionResponse,
        )

    @param_model(ListSessionsRequest)
    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        return await request_model_from_dict(
            self._conn,
            AGENT_METHODS["session_list"],
            ListSessionsRequest(cursor=cursor, cwd=cwd, field_meta=kwargs or None),
            ListSessionsResponse,
        )

    @param_model(SetSessionModeRequest)
    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> SetSessionModeResponse:
        return await request_model_from_dict(
            self._conn,
            AGENT_METHODS["session_set_mode"],
            SetSessionModeRequest(mode_id=mode_id, session_id=session_id, field_meta=kwargs or None),
            SetSessionModeResponse,
        )

    @param_model(SetSessionConfigOptionBooleanRequest | SetSessionConfigOptionSelectRequest)
    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any
    ) -> SetSessionConfigOptionResponse:
        request = (
            SetSessionConfigOptionBooleanRequest(
                config_id=config_id, session_id=session_id, type="boolean", value=value, field_meta=kwargs or None
            )
            if isinstance(value, bool)
            else SetSessionConfigOptionSelectRequest(
                config_id=config_id, session_id=session_id, value=value, field_meta=kwargs or None
            )
        )
        return await request_model_from_dict(
            self._conn, AGENT_METHODS["session_set_config_option"], request, SetSessionConfigOptionResponse
        )

    @param_model(AuthenticateRequest)
    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse:
        return await request_model_from_dict(
            self._conn,
            AGENT_METHODS["authenticate"],
            AuthenticateRequest(method_id=method_id, field_meta=kwargs or None),
            AuthenticateResponse,
        )

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
    ) -> PromptResponse:
        try:
            response = await request_model(
                self._conn,
                AGENT_METHODS["session_prompt"],
                PromptRequest(prompt=prompt, session_id=session_id, field_meta=kwargs or None),
                PromptResponse,
            )
        except Exception:
            await self._session_updates.wait(session_id)
            raise
        await self._session_updates.wait(session_id)
        return response

    @param_model(ForkSessionRequest)
    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        return await request_model(
            self._conn,
            AGENT_METHODS["session_fork"],
            ForkSessionRequest(
                session_id=session_id,
                cwd=cwd,
                additional_directories=additional_directories,
                mcp_servers=mcp_servers,
                field_meta=kwargs or None,
            ),
            ForkSessionResponse,
        )

    @param_model(ResumeSessionRequest)
    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        return await request_model(
            self._conn,
            AGENT_METHODS["session_resume"],
            ResumeSessionRequest(
                session_id=session_id,
                cwd=cwd,
                additional_directories=additional_directories,
                mcp_servers=mcp_servers,
                field_meta=kwargs or None,
            ),
            ResumeSessionResponse,
        )

    @param_model(CloseSessionRequest)
    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse | None:
        return await request_model_from_dict(
            self._conn,
            AGENT_METHODS["session_close"],
            CloseSessionRequest(session_id=session_id, field_meta=kwargs or None),
            CloseSessionResponse,
        )

    @param_model(CancelNotification)
    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        await notify_model(
            self._conn,
            AGENT_METHODS["session_cancel"],
            CancelNotification(session_id=session_id, field_meta=kwargs or None),
        )

    @param_model(DeleteSessionRequest)
    async def delete_session(self, session_id: str, **kwargs: Any) -> DeleteSessionResponse:
        return await request_model_from_dict(
            self._conn,
            AGENT_METHODS["session_delete"],
            DeleteSessionRequest(session_id=session_id, field_meta=kwargs or None),
            DeleteSessionResponse,
        )

    @param_model(ListProvidersRequest)
    async def list_providers(self, **kwargs: Any) -> ListProvidersResponse:
        return await request_model(
            self._conn,
            AGENT_METHODS["providers_list"],
            ListProvidersRequest(field_meta=kwargs or None),
            ListProvidersResponse,
        )

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
    ) -> SetProviderResponse:
        return await request_model_from_dict(
            self._conn,
            AGENT_METHODS["providers_set"],
            SetProviderRequest(
                provider_id=provider_id,
                api_type=api_type,
                base_url=base_url,
                headers=headers,
                field_meta=kwargs or None,
            ),
            SetProviderResponse,
        )

    @param_model(DisableProviderRequest)
    async def disable_provider(self, provider_id: str, **kwargs: Any) -> DisableProviderResponse:
        return await request_model_from_dict(
            self._conn,
            AGENT_METHODS["providers_disable"],
            DisableProviderRequest(provider_id=provider_id, field_meta=kwargs or None),
            DisableProviderResponse,
        )

    @param_model(LogoutRequest)
    async def logout(self, **kwargs: Any) -> LogoutResponse:
        return await request_model_from_dict(
            self._conn, AGENT_METHODS["logout"], LogoutRequest(field_meta=kwargs or None), LogoutResponse
        )

    @param_model(MessageMcpRequest)
    async def mcp_message(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        return await self._conn.send_request(
            AGENT_METHODS["mcp_message"],
            serialize_params(
                MessageMcpRequest(connection_id=connection_id, method=method, params=params, field_meta=kwargs or None)
            ),
        )

    @param_model(MessageMcpNotification)
    async def notify_mcp(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        await notify_model(
            self._conn,
            AGENT_METHODS["mcp_message"],
            MessageMcpNotification(
                connection_id=connection_id, method=method, params=params, field_meta=kwargs or None
            ),
        )

    @param_model(StartNesRequest)
    async def start_nes(
        self,
        workspace_uri: str | None = None,
        workspace_folders: list[WorkspaceFolder] | None = None,
        repository: NesRepository | None = None,
        **kwargs: Any,
    ) -> StartNesResponse:
        return await request_model(
            self._conn,
            AGENT_METHODS["nes_start"],
            StartNesRequest(
                workspace_uri=workspace_uri,
                workspace_folders=workspace_folders,
                repository=repository,
                field_meta=kwargs or None,
            ),
            StartNesResponse,
        )

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
    ) -> SuggestNesResponse:
        return await request_model(
            self._conn,
            AGENT_METHODS["nes_suggest"],
            SuggestNesRequest(
                session_id=session_id,
                uri=uri,
                version=version,
                position=position,
                selection=selection,
                trigger_kind=trigger_kind,
                context=context,
                field_meta=kwargs or None,
            ),
            SuggestNesResponse,
        )

    @param_model(CloseNesRequest)
    async def close_nes(self, session_id: str, **kwargs: Any) -> CloseNesResponse:
        return await request_model_from_dict(
            self._conn,
            AGENT_METHODS["nes_close"],
            CloseNesRequest(session_id=session_id, field_meta=kwargs or None),
            CloseNesResponse,
        )

    @param_model(AcceptNesNotification)
    async def accept_nes(self, session_id: str, id: str, **kwargs: Any) -> None:  # noqa: A002
        await notify_model(
            self._conn,
            AGENT_METHODS["nes_accept"],
            AcceptNesNotification(session_id=session_id, id=id, field_meta=kwargs or None),
        )

    @param_model(RejectNesNotification)
    async def reject_nes(
        self,
        session_id: str,
        id: str,  # noqa: A002
        reason: Literal["rejected", "ignored", "replaced", "cancelled"] | None = None,
        **kwargs: Any,
    ) -> None:
        await notify_model(
            self._conn,
            AGENT_METHODS["nes_reject"],
            RejectNesNotification(session_id=session_id, id=id, reason=reason, field_meta=kwargs or None),
        )

    @param_model(DidOpenDocumentNotification)
    async def did_open(
        self, session_id: str, uri: str, language_id: str, version: int, text: str, **kwargs: Any
    ) -> None:
        await notify_model(
            self._conn,
            AGENT_METHODS["document_did_open"],
            DidOpenDocumentNotification(
                session_id=session_id,
                uri=uri,
                language_id=language_id,
                version=version,
                text=text,
                field_meta=kwargs or None,
            ),
        )

    @param_model(DidChangeDocumentNotification)
    async def did_change(
        self,
        session_id: str,
        uri: str,
        version: int,
        content_changes: list[TextDocumentContentChangeEvent],
        **kwargs: Any,
    ) -> None:
        await notify_model(
            self._conn,
            AGENT_METHODS["document_did_change"],
            DidChangeDocumentNotification(
                session_id=session_id,
                uri=uri,
                version=version,
                content_changes=content_changes,
                field_meta=kwargs or None,
            ),
        )

    @param_model(DidCloseDocumentNotification)
    async def did_close(self, session_id: str, uri: str, **kwargs: Any) -> None:
        await notify_model(
            self._conn,
            AGENT_METHODS["document_did_close"],
            DidCloseDocumentNotification(session_id=session_id, uri=uri, field_meta=kwargs or None),
        )

    @param_model(DidSaveDocumentNotification)
    async def did_save(self, session_id: str, uri: str, **kwargs: Any) -> None:
        await notify_model(
            self._conn,
            AGENT_METHODS["document_did_save"],
            DidSaveDocumentNotification(session_id=session_id, uri=uri, field_meta=kwargs or None),
        )

    @param_model(DidFocusDocumentNotification)
    async def did_focus(
        self, session_id: str, uri: str, version: int, position: Position, visible_range: Range, **kwargs: Any
    ) -> None:
        await notify_model(
            self._conn,
            AGENT_METHODS["document_did_focus"],
            DidFocusDocumentNotification(
                session_id=session_id,
                uri=uri,
                version=version,
                position=position,
                visible_range=visible_range,
                field_meta=kwargs or None,
            ),
        )

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._conn.send_request(f"_{method}", params)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        await self._conn.send_notification(f"_{method}", params)

    async def close(self) -> None:
        await self._conn.close()

    async def __aenter__(self) -> ClientSideConnection:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def on_connect(self, conn: Client) -> None:
        pass
