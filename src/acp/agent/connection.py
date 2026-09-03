from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast, final

from pydantic import TypeAdapter

from .._transport import Transport
from ..connection import Connection, MethodHandler
from ..interfaces import Agent, Client
from ..meta import CLIENT_METHODS
from ..schema import (
    AcceptElicitationResponse,
    AgentMessageChunk,
    AgentPlanContentUpdate,
    AgentPlanRemovedUpdate,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AvailableCommandsUpdate,
    CancelElicitationResponse,
    CompleteElicitationNotification,
    ConfigOptionUpdate,
    CreateElicitationResponse,
    CreateFormElicitationRequest,
    CreateFormRequestElicitationRequest,
    CreateFormSessionElicitationRequest,
    CreateTerminalRequest,
    CreateTerminalResponse,
    CreateUrlElicitationRequest,
    CreateUrlRequestElicitationRequest,
    CreateUrlSessionElicitationRequest,
    CurrentModeUpdate,
    DeclineElicitationResponse,
    ElicitationFormRequestMode,
    ElicitationFormSessionMode,
    ElicitationMode,
    ElicitationUrlRequestMode,
    ElicitationUrlSessionMode,
    EnvVariable,
    KillTerminalRequest,
    KillTerminalResponse,
    PermissionOption,
    ReadTextFileRequest,
    ReadTextFileResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
    SessionInfoUpdate,
    SessionNotification,
    SessionUpdateCompactionSummaryChunk,
    SessionUpdateCompactionUpdate,
    TerminalOutputRequest,
    TerminalOutputResponse,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UsageUpdate,
    UserMessageChunk,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
    WriteTextFileRequest,
    WriteTextFileResponse,
)
from ..utils import compatible_class, notify_model, param_model, request_model, request_optional_model, serialize_params
from .router import build_agent_router

__all__ = ["AgentSideConnection"]
_AGENT_CONNECTION_ERROR = "AgentSideConnection requires asyncio StreamWriter/StreamReader"
_CREATE_ELICITATION_RESPONSE_ADAPTER = TypeAdapter(CreateElicitationResponse)


@final
@compatible_class
class AgentSideConnection:
    """Agent-side connection wrapper that dispatches JSON-RPC messages to a Client implementation.
    The agent can use this connection to communicate with the Client so it behaves like a Client.
    """

    def __init__(
        self,
        to_agent: Callable[[Client], Agent] | Agent,
        input_stream: Any,
        output_stream: Any = None,
        listening: bool = True,
        *,
        use_unstable_protocol: bool = False,
        **connection_kwargs: Any,
    ) -> None:
        agent, handler = self._prepare(to_agent, use_unstable_protocol=use_unstable_protocol)
        if isinstance(input_stream, Transport):
            if output_stream is not None:
                raise TypeError(_AGENT_CONNECTION_ERROR)
            self._conn = Connection(handler, input_stream, listening=listening, **connection_kwargs)
        else:
            if not isinstance(input_stream, asyncio.StreamWriter) or not isinstance(
                output_stream, asyncio.StreamReader
            ):
                raise TypeError(_AGENT_CONNECTION_ERROR)
            self._conn = Connection(handler, input_stream, output_stream, listening=listening, **connection_kwargs)
        self._notify_connected(agent)

    @classmethod
    def attach(
        cls,
        to_agent: Callable[[Client], Agent] | Agent,
        connection: Connection,
        *,
        use_unstable_protocol: bool = False,
    ) -> tuple[AgentSideConnection, MethodHandler]:
        """Attach an agent-side wrapper to an existing connection."""
        self = cls.__new__(cls)
        agent, handler = self._prepare(to_agent, use_unstable_protocol=use_unstable_protocol)
        self._conn = connection
        self._notify_connected(agent)
        return self, handler

    def _prepare(
        self,
        to_agent: Callable[[Client], Agent] | Agent,
        *,
        use_unstable_protocol: bool,
    ) -> tuple[Agent, MethodHandler]:
        agent = cast(Agent, to_agent(self) if callable(to_agent) else to_agent)
        return agent, build_agent_router(agent, use_unstable_protocol=use_unstable_protocol)

    def _notify_connected(self, agent: Agent) -> None:
        if on_connect := getattr(agent, "on_connect", None):
            on_connect(self)

    async def listen(self) -> None:
        """Start listening for incoming messages."""
        await self._conn.main_loop()

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
        | SessionUpdateCompactionUpdate
        | SessionUpdateCompactionSummaryChunk,
        **kwargs: Any,
    ) -> None:
        await notify_model(
            self._conn,
            CLIENT_METHODS["session_update"],
            SessionNotification(session_id=session_id, update=update, field_meta=kwargs or None),
        )

    @param_model(RequestPermissionRequest)
    async def request_permission(
        self, session_id: str, tool_call: ToolCallUpdate, options: list[PermissionOption], **kwargs: Any
    ) -> RequestPermissionResponse:
        return await request_model(
            self._conn,
            CLIENT_METHODS["session_request_permission"],
            RequestPermissionRequest(
                options=options, session_id=session_id, tool_call=tool_call, field_meta=kwargs or None
            ),
            RequestPermissionResponse,
        )

    @param_model(ReadTextFileRequest)
    async def read_text_file(
        self, session_id: str, path: str, line: int | None = None, limit: int | None = None, **kwargs: Any
    ) -> ReadTextFileResponse:
        return await request_model(
            self._conn,
            CLIENT_METHODS["fs_read_text_file"],
            ReadTextFileRequest(path=path, session_id=session_id, limit=limit, line=line, field_meta=kwargs or None),
            ReadTextFileResponse,
        )

    @param_model(WriteTextFileRequest)
    async def write_text_file(
        self, session_id: str, path: str, content: str, **kwargs: Any
    ) -> WriteTextFileResponse | None:
        return await request_optional_model(
            self._conn,
            CLIENT_METHODS["fs_write_text_file"],
            WriteTextFileRequest(content=content, path=path, session_id=session_id, field_meta=kwargs or None),
            WriteTextFileResponse,
        )

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
    ) -> CreateTerminalResponse:
        return await request_model(
            self._conn,
            CLIENT_METHODS["terminal_create"],
            CreateTerminalRequest(
                command=command,
                session_id=session_id,
                args=args,
                cwd=cwd,
                env=env,
                output_byte_limit=output_byte_limit,
                field_meta=kwargs or None,
            ),
            CreateTerminalResponse,
        )

    @param_model(TerminalOutputRequest)
    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse:
        return await request_model(
            self._conn,
            CLIENT_METHODS["terminal_output"],
            TerminalOutputRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None),
            TerminalOutputResponse,
        )

    @param_model(ReleaseTerminalRequest)
    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        return await request_optional_model(
            self._conn,
            CLIENT_METHODS["terminal_release"],
            ReleaseTerminalRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None),
            ReleaseTerminalResponse,
        )

    @param_model(WaitForTerminalExitRequest)
    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        return await request_model(
            self._conn,
            CLIENT_METHODS["terminal_wait_for_exit"],
            WaitForTerminalExitRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None),
            WaitForTerminalExitResponse,
        )

    @param_model(KillTerminalRequest)
    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> KillTerminalResponse | None:
        return await request_optional_model(
            self._conn,
            CLIENT_METHODS["terminal_kill"],
            KillTerminalRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None),
            KillTerminalResponse,
        )

    async def create_elicitation(
        self, message: str, mode: ElicitationMode, **kwargs: Any
    ) -> AcceptElicitationResponse | DeclineElicitationResponse | CancelElicitationResponse:
        request = _create_elicitation_request(message, mode, kwargs or None)
        response = await self._conn.send_request(CLIENT_METHODS["elicitation_create"], serialize_params(request))
        return _CREATE_ELICITATION_RESPONSE_ADAPTER.validate_python(response)

    @param_model(CompleteElicitationNotification)
    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        await notify_model(
            self._conn,
            CLIENT_METHODS["elicitation_complete"],
            CompleteElicitationNotification(elicitation_id=elicitation_id, field_meta=kwargs or None),
        )

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._conn.send_request(f"_{method}", params)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        await self._conn.send_notification(f"_{method}", params)

    async def close(self) -> None:
        await self._conn.close()

    async def __aenter__(self) -> AgentSideConnection:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def on_connect(self, conn: Agent) -> None:
        pass


def _create_elicitation_request(
    message: str, mode: ElicitationMode, field_meta: dict[str, Any] | None
) -> CreateFormElicitationRequest | CreateUrlElicitationRequest:
    mode_fields = mode.model_dump(mode="json", exclude_none=True)
    if isinstance(mode, ElicitationFormSessionMode):
        return CreateFormSessionElicitationRequest(message=message, mode="form", field_meta=field_meta, **mode_fields)
    if isinstance(mode, ElicitationFormRequestMode):
        return CreateFormRequestElicitationRequest(message=message, mode="form", field_meta=field_meta, **mode_fields)
    if isinstance(mode, ElicitationUrlSessionMode):
        return CreateUrlSessionElicitationRequest(message=message, mode="url", field_meta=field_meta, **mode_fields)
    if isinstance(mode, ElicitationUrlRequestMode):
        return CreateUrlRequestElicitationRequest(message=message, mode="url", field_meta=field_meta, **mode_fields)
    raise TypeError(f"Unsupported elicitation mode: {type(mode).__name__}")
