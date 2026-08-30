from typing import Any

from .core import (
    Agent,
    Client,
    RequestError,
    connect_to_agent,
    run_agent,
)
from .meta import (
    AGENT_METHODS,
    CLIENT_METHODS,
    PROTOCOL_VERSION,
)
from .schema import (
    AcceptElicitationResponse,
    AuthenticateRequest,
    AuthenticateResponse,
    CancelElicitationResponse,
    CancelNotification,
    CompleteElicitationNotification,
    CreateElicitationRequest,
    CreateElicitationResponse,
    CreateFormElicitationRequest,
    CreateFormRequestElicitationRequest,
    CreateFormSessionElicitationRequest,
    CreateOtherElicitationRequest,
    CreateTerminalRequest,
    CreateTerminalResponse,
    CreateUrlElicitationRequest,
    CreateUrlRequestElicitationRequest,
    CreateUrlSessionElicitationRequest,
    DeclineElicitationResponse,
    ElicitationBooleanPropertySchema,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
    ElicitationFormRequestMode,
    ElicitationFormSessionMode,
    ElicitationIntegerPropertySchema,
    ElicitationMode,
    ElicitationMultiSelectPropertySchema,
    ElicitationNumberPropertySchema,
    ElicitationOtherPropertySchema,
    ElicitationSchema,
    ElicitationStringPropertySchema,
    ElicitationUrlCapabilities,
    ElicitationUrlRequestMode,
    ElicitationUrlSessionMode,
    InitializeRequest,
    InitializeResponse,
    KillTerminalRequest,
    KillTerminalResponse,
    LoadSessionRequest,
    LoadSessionResponse,
    NewSessionRequest,
    NewSessionResponse,
    OtherElicitationResponse,
    PromptRequest,
    PromptResponse,
    ReadTextFileRequest,
    ReadTextFileResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
    SessionNotification,
    SetSessionConfigOptionResponse,
    SetSessionConfigOptionSelectRequest,
    SetSessionModeRequest,
    SetSessionModeResponse,
    TerminalOutputRequest,
    TerminalOutputResponse,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
    WriteTextFileRequest,
    WriteTextFileResponse,
)
from .stdio import spawn_agent_process, spawn_client_process, spawn_stdio_connection, stdio_streams
from .transports import default_environment, spawn_stdio_transport

_DEPRECATED_NAMES = [
    (
        "AgentSideConnection",
        "acp.core:AgentSideConnection",
        "Using `AgentSideConnection` directly is deprecated, please use `acp.run_agent` instead.",
    ),
    (
        "ClientSideConnection",
        "acp.core:ClientSideConnection",
        "Using `ClientSideConnection` directly is deprecated, please use `acp.connect_to_agent` instead.",
    ),
]

__all__ = [  # noqa: RUF022
    # constants
    "PROTOCOL_VERSION",
    "AGENT_METHODS",
    "CLIENT_METHODS",
    # types
    "InitializeRequest",
    "InitializeResponse",
    "NewSessionRequest",
    "NewSessionResponse",
    "LoadSessionRequest",
    "LoadSessionResponse",
    "AuthenticateRequest",
    "AuthenticateResponse",
    "PromptRequest",
    "PromptResponse",
    "WriteTextFileRequest",
    "WriteTextFileResponse",
    "ReadTextFileRequest",
    "ReadTextFileResponse",
    "RequestPermissionRequest",
    "RequestPermissionResponse",
    "CancelNotification",
    "SessionNotification",
    "SetSessionModeRequest",
    "SetSessionModeResponse",
    "SetSessionConfigOptionSelectRequest",
    "SetSessionConfigOptionResponse",
    # elicitation types
    "ElicitationMode",
    "ElicitationSchema",
    "ElicitationCapabilities",
    "ElicitationFormCapabilities",
    "ElicitationUrlCapabilities",
    "ElicitationFormSessionMode",
    "ElicitationFormRequestMode",
    "ElicitationUrlSessionMode",
    "ElicitationUrlRequestMode",
    "ElicitationStringPropertySchema",
    "ElicitationNumberPropertySchema",
    "ElicitationIntegerPropertySchema",
    "ElicitationBooleanPropertySchema",
    "ElicitationMultiSelectPropertySchema",
    "ElicitationOtherPropertySchema",
    "CreateElicitationRequest",
    "CreateElicitationResponse",
    "CreateFormElicitationRequest",
    "CreateFormSessionElicitationRequest",
    "CreateFormRequestElicitationRequest",
    "CreateUrlElicitationRequest",
    "CreateUrlSessionElicitationRequest",
    "CreateUrlRequestElicitationRequest",
    "CreateOtherElicitationRequest",
    "AcceptElicitationResponse",
    "DeclineElicitationResponse",
    "CancelElicitationResponse",
    "OtherElicitationResponse",
    "CompleteElicitationNotification",
    # terminal types
    "CreateTerminalRequest",
    "CreateTerminalResponse",
    "TerminalOutputRequest",
    "TerminalOutputResponse",
    "WaitForTerminalExitRequest",
    "WaitForTerminalExitResponse",
    "KillTerminalRequest",
    "KillTerminalResponse",
    "ReleaseTerminalRequest",
    "ReleaseTerminalResponse",
    # core
    "run_agent",
    "connect_to_agent",
    "RequestError",
    "Agent",
    "Client",
    # stdio helper
    "stdio_streams",
    "spawn_stdio_connection",
    "spawn_agent_process",
    "spawn_client_process",
    "default_environment",
    "spawn_stdio_transport",
]


def __getattr__(name: str) -> Any:
    import warnings
    from importlib import import_module

    for deprecated_name, new_path, warning in _DEPRECATED_NAMES:
        if name == deprecated_name:
            warnings.warn(warning, DeprecationWarning, stacklevel=2)
            module_name, attr_name = new_path.split(":")
            module = import_module(module_name)
            return getattr(module, attr_name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
