from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from . import schema
from .meta import AGENT_METHODS, CLIENT_METHODS


@dataclass(frozen=True, slots=True)
class RequestSpec:
    method: str
    handler: str
    request: TypeAdapter[Any]
    response: TypeAdapter[Any]
    empty_response: bool = False


@dataclass(frozen=True, slots=True)
class NotificationSpec:
    method: str
    handler: str
    params: TypeAdapter[Any]


def request(
    method: str,
    handler: str,
    request_type: Any,
    response_type: Any,
    *,
    empty_response: bool = False,
) -> RequestSpec:
    return RequestSpec(
        method=method,
        handler=handler,
        request=TypeAdapter(request_type),
        response=TypeAdapter(response_type),
        empty_response=empty_response,
    )


def notification(method: str, handler: str, params_type: Any) -> NotificationSpec:
    return NotificationSpec(method=method, handler=handler, params=TypeAdapter(params_type))


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


AGENT_REQUESTS = (
    request(AGENT_METHODS["initialize"], "initialize", schema.InitializeRequest, schema.InitializeResponse),
    request(
        AGENT_METHODS["auth_login"],
        "login",
        schema.LoginAuthRequest,
        schema.LoginAuthResponse,
        empty_response=True,
    ),
    request(
        AGENT_METHODS["providers_list"], "list_providers", schema.ListProvidersRequest, schema.ListProvidersResponse
    ),
    request(
        AGENT_METHODS["providers_set"],
        "set_provider",
        schema.SetProviderRequest,
        schema.SetProviderResponse,
        empty_response=True,
    ),
    request(
        AGENT_METHODS["providers_disable"],
        "disable_provider",
        schema.DisableProviderRequest,
        schema.DisableProviderResponse,
        empty_response=True,
    ),
    request(AGENT_METHODS["session_new"], "new_session", schema.NewSessionRequest, schema.NewSessionResponse),
    request(
        AGENT_METHODS["session_set_config_option"],
        "set_config_option",
        SetConfigOptionRequest,
        schema.SetSessionConfigOptionResponse,
    ),
    request(
        AGENT_METHODS["session_prompt"],
        "prompt",
        schema.PromptRequest,
        schema.PromptResponse,
        empty_response=True,
    ),
    request(AGENT_METHODS["mcp_message"], "message_mcp", schema.MessageMcpRequest, Any),
    request(AGENT_METHODS["session_list"], "list_sessions", schema.ListSessionsRequest, schema.ListSessionsResponse),
    request(
        AGENT_METHODS["session_delete"],
        "delete_session",
        schema.DeleteSessionRequest,
        schema.DeleteSessionResponse,
        empty_response=True,
    ),
    request(AGENT_METHODS["session_fork"], "fork_session", schema.ForkSessionRequest, schema.ForkSessionResponse),
    request(
        AGENT_METHODS["session_resume"], "resume_session", schema.ResumeSessionRequest, schema.ResumeSessionResponse
    ),
    request(
        AGENT_METHODS["session_close"],
        "close_session",
        schema.CloseSessionRequest,
        schema.CloseSessionResponse,
        empty_response=True,
    ),
    request(
        AGENT_METHODS["auth_logout"],
        "logout",
        schema.LogoutAuthRequest,
        schema.LogoutAuthResponse,
        empty_response=True,
    ),
    request(AGENT_METHODS["nes_start"], "start_nes", schema.StartNesRequest, schema.StartNesResponse),
    request(AGENT_METHODS["nes_suggest"], "suggest_nes", schema.SuggestNesRequest, schema.SuggestNesResponse),
    request(
        AGENT_METHODS["nes_close"],
        "close_nes",
        schema.CloseNesRequest,
        schema.CloseNesResponse,
        empty_response=True,
    ),
)

AGENT_NOTIFICATIONS = (
    notification(AGENT_METHODS["session_cancel"], "cancel", schema.CancelSessionNotification),
    notification(AGENT_METHODS["mcp_message"], "message_mcp", schema.MessageMcpNotification),
    notification(AGENT_METHODS["document_did_open"], "did_open", schema.DidOpenDocumentNotification),
    notification(AGENT_METHODS["document_did_change"], "did_change", schema.DidChangeDocumentNotification),
    notification(AGENT_METHODS["document_did_close"], "did_close", schema.DidCloseDocumentNotification),
    notification(AGENT_METHODS["document_did_save"], "did_save", schema.DidSaveDocumentNotification),
    notification(AGENT_METHODS["document_did_focus"], "did_focus", schema.DidFocusDocumentNotification),
    notification(AGENT_METHODS["nes_accept"], "accept_nes", schema.AcceptNesNotification),
    notification(AGENT_METHODS["nes_reject"], "reject_nes", schema.RejectNesNotification),
)

CLIENT_REQUESTS = (
    request(
        CLIENT_METHODS["session_request_permission"],
        "request_permission",
        schema.RequestPermissionRequest,
        schema.RequestPermissionResponse,
    ),
    request(CLIENT_METHODS["mcp_connect"], "connect_mcp", schema.ConnectMcpRequest, schema.ConnectMcpResponse),
    request(CLIENT_METHODS["mcp_message"], "message_mcp", schema.MessageMcpRequest, Any),
    request(
        CLIENT_METHODS["mcp_disconnect"],
        "disconnect_mcp",
        schema.DisconnectMcpRequest,
        schema.DisconnectMcpResponse,
        empty_response=True,
    ),
    request(
        CLIENT_METHODS["elicitation_create"],
        "create_elicitation",
        CreateElicitationRequest,
        CreateElicitationResponse,
    ),
)

CLIENT_NOTIFICATIONS = (
    notification(CLIENT_METHODS["session_update"], "session_update", schema.UpdateSessionNotification),
    notification(CLIENT_METHODS["mcp_message"], "message_mcp", schema.MessageMcpNotification),
    notification(
        CLIENT_METHODS["elicitation_complete"],
        "complete_elicitation",
        schema.CompleteElicitationNotification,
    ),
)


AGENT_REQUESTS_BY_METHOD = {spec.method: spec for spec in AGENT_REQUESTS}
AGENT_NOTIFICATIONS_BY_METHOD = {spec.method: spec for spec in AGENT_NOTIFICATIONS}
CLIENT_REQUESTS_BY_METHOD = {spec.method: spec for spec in CLIENT_REQUESTS}
CLIENT_NOTIFICATIONS_BY_METHOD = {spec.method: spec for spec in CLIENT_NOTIFICATIONS}
