# Generated from schema/schema.json. Do not edit by hand.
# Schema ref: refs/tags/schema-v1.19.0

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import AnyUrl, BaseModel as _BaseModel, ConfigDict, Field, RootModel, field_validator
from acp._deserialize import salvage_on_error, skip_invalid_items

PermissionOptionKind = Literal["allow_once", "allow_always", "reject_once", "reject_always"]
PlanEntryPriority = Literal["high", "medium", "low"]
PlanEntryStatus = Literal["pending", "in_progress", "completed"]
StopReason = Literal["end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"]
ToolCallStatus = Literal["pending", "in_progress", "completed", "failed"]
ToolKind = Literal["read", "edit", "delete", "move", "search", "execute", "think", "fetch", "switch_mode", "other"]


class BaseModel(_BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_attribute_docstrings=True)

    def __getattr__(self, item: str) -> Any:
        if item.lower() != item:
            snake_cased = "".join("_" + c.lower() if c.isupper() and i > 0 else c.lower() for i, c in enumerate(item))
            return getattr(self, snake_cased)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    @field_validator("field_meta", mode="wrap", check_fields=False)
    @classmethod
    def _salvage_meta_on_error(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class Jsonrpc(Enum):
    field_2_0 = "2.0"


class ReadTextFileRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this request.
    """
    path: str
    """
    Absolute path to the file to read.
    """
    line: Annotated[Optional[int], Field(ge=0)] = None
    """
    Line number to start reading from (1-based).
    """
    limit: Annotated[Optional[int], Field(ge=0)] = None
    """
    Maximum number of lines to read.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("limit", "line", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class TextResourceContents(BaseModel):
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type describing the encoded media payload.
    """
    text: str
    """
    Text payload carried by this content block.
    """
    uri: str
    """
    URI associated with this resource or media payload.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("mime_type", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class BlobResourceContents(BaseModel):
    blob: str
    """
    Base64-encoded bytes for a binary resource payload.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type describing the encoded media payload.
    """
    uri: str
    """
    URI associated with this resource or media payload.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("mime_type", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class Diff(BaseModel):
    path: str
    """
    The absolute file path being modified.
    """
    old_text: Annotated[Optional[str], Field(alias="oldText")] = None
    """
    The original content (None for new files).
    """
    new_text: Annotated[str, Field(alias="newText")]
    """
    The new content after modification.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("old_text", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class Terminal(BaseModel):
    terminal_id: Annotated[str, Field(alias="terminalId")]
    """
    Identifier of the terminal instance to embed in the content stream.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ToolCallLocation(BaseModel):
    path: str
    """
    The absolute file path being accessed or modified.
    """
    line: Annotated[Optional[int], Field(ge=0)] = None
    """
    Optional line number within the file.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("line", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class EnvVariable(BaseModel):
    name: str
    """
    The name of the environment variable.
    """
    value: str
    """
    The value to set for the environment variable.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class TerminalOutputRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this request.
    """
    terminal_id: Annotated[str, Field(alias="terminalId")]
    """
    The ID of the terminal to get output from.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ReleaseTerminalRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this request.
    """
    terminal_id: Annotated[str, Field(alias="terminalId")]
    """
    The ID of the terminal to release.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class WaitForTerminalExitRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this request.
    """
    terminal_id: Annotated[str, Field(alias="terminalId")]
    """
    The ID of the terminal to wait for.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class KillTerminalRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this request.
    """
    terminal_id: Annotated[str, Field(alias="terminalId")]
    """
    The ID of the terminal to kill.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class CreateOtherElicitationRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    message: str
    """
    A human-readable message describing what input is needed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """
    mode: str
    """
    Custom or future elicitation mode.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """

    @field_validator("mode", mode="before")
    @classmethod
    def _reject_known_mode(cls, value: Any) -> Any:
        # Restore the schema's `not` clause dropped for codegen: reject the known
        # variants' discriminator values so a malformed known variant fails instead
        # of silently parsing as this catch-all.
        if value in ("form", "url"):
            raise ValueError("mode value is reserved by a known variant")
        return value


class ElicitationSessionScope(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session this elicitation is tied to.
    """
    tool_call_id: Annotated[Optional[str], Field(alias="toolCallId")] = None
    """
    Optional tool call within the session.
    """

    @field_validator("tool_call_id", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ElicitationRequestScope(BaseModel):
    request_id: Annotated[Optional[Union[int, str]], Field(alias="requestId")]
    """
    The request this elicitation is tied to.
    """


class ElicitationOtherPropertySchema(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: str
    """
    Custom or future elicitation property schema type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """

    @field_validator("type", mode="before")
    @classmethod
    def _reject_known_type(cls, value: Any) -> Any:
        # Restore the schema's `not` clause dropped for codegen: reject the known
        # variants' discriminator values so a malformed known variant fails instead
        # of silently parsing as this catch-all.
        if value in ("string", "number", "integer", "boolean", "array"):
            raise ValueError("type value is reserved by a known variant")
        return value


class EnumOption(BaseModel):
    const: str
    """
    The constant value for this option.
    """
    title: str
    """
    Human-readable title for this option.
    """
    description: Optional[str] = None
    """
    Human-readable description.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class StringPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional title for the property.
    """
    description: Optional[str] = None
    """
    Human-readable description.
    """
    min_length: Annotated[Optional[int], Field(alias="minLength", ge=0)] = None
    """
    Minimum string length.
    """
    max_length: Annotated[Optional[int], Field(alias="maxLength", ge=0)] = None
    """
    Maximum string length.
    """
    pattern: Optional[str] = None
    """
    Pattern the string must match.
    """
    format: Optional[str] = None
    """
    String format.
    """
    default: Optional[str] = None
    """
    Default value.
    """
    enum: Optional[List[str]] = None
    """
    Enum values for untitled single-select enums.
    """
    one_of: Annotated[Optional[List[EnumOption]], Field(alias="oneOf")] = None
    """
    Titled enum options for titled single-select enums.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NumberPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional title for the property.
    """
    description: Optional[str] = None
    """
    Human-readable description.
    """
    minimum: Optional[float] = None
    """
    Minimum value (inclusive).
    """
    maximum: Optional[float] = None
    """
    Maximum value (inclusive).
    """
    default: Optional[float] = None
    """
    Default value.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class IntegerPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional title for the property.
    """
    description: Optional[str] = None
    """
    Human-readable description.
    """
    minimum: Optional[int] = None
    """
    Minimum value (inclusive).
    """
    maximum: Optional[int] = None
    """
    Maximum value (inclusive).
    """
    default: Optional[int] = None
    """
    Default value.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class BooleanPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional title for the property.
    """
    description: Optional[str] = None
    """
    Human-readable description.
    """
    default: Optional[bool] = None
    """
    Default value.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class OtherMultiSelectItems(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: str
    """
    Custom or future multi-select item type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """

    @field_validator("type", mode="before")
    @classmethod
    def _reject_known_type(cls, value: Any) -> Any:
        # Restore the schema's `not` clause dropped for codegen: reject the known
        # variants' discriminator values so a malformed known variant fails instead
        # of silently parsing as this catch-all.
        if value in ("string",):
            raise ValueError("type value is reserved by a known variant")
        return value


class _StringMultiSelectItems(BaseModel):
    enum: List[str]
    """
    Allowed enum values.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class TitledMultiSelectItems(BaseModel):
    any_of: Annotated[List[EnumOption], Field(alias="anyOf")]
    """
    Titled enum options.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ElicitationUrlSessionMode(ElicitationSessionScope):
    elicitation_id: Annotated[str, Field(alias="elicitationId")]
    """
    The unique identifier for this elicitation.
    """
    url: AnyUrl
    """
    The URL to direct the user to.
    """


class ElicitationUrlRequestMode(ElicitationRequestScope):
    elicitation_id: Annotated[str, Field(alias="elicitationId")]
    """
    The unique identifier for this elicitation.
    """
    url: AnyUrl
    """
    The URL to direct the user to.
    """


class ElicitationUrlMode(RootModel[Union[ElicitationUrlSessionMode, ElicitationUrlRequestMode]]):
    model_config = ConfigDict(use_attribute_docstrings=True)

    root: Union[ElicitationUrlSessionMode, ElicitationUrlRequestMode]
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    URL-based elicitation mode where the client directs the user to a URL.
    """


class DisconnectMcpRequest(BaseModel):
    connection_id: Annotated[str, Field(alias="connectionId")]
    """
    The MCP-over-ACP connection to close.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class PromptCapabilities(BaseModel):
    image: Optional[bool] = False
    """
    Agent supports [`ContentBlock::Image`].
    """
    audio: Optional[bool] = False
    """
    Agent supports [`ContentBlock::Audio`].
    """
    embedded_context: Annotated[Optional[bool], Field(alias="embeddedContext")] = False
    """
    Agent supports embedded context in `session/prompt` requests.

    When enabled, the Client is allowed to include [`ContentBlock::Resource`]
    in prompt requests for pieces of context that are referenced in the message.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("audio", "embedded_context", "image", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: False)


class McpCapabilities(BaseModel):
    http: Optional[bool] = False
    """
    Agent supports [`McpServer::Http`].
    """
    sse: Optional[bool] = False
    """
    Agent supports [`McpServer::Sse`].
    """
    acp: Optional[bool] = False
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Agent supports [`McpServer::Acp`].
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("acp", "http", "sse", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: False)


class SessionListCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class SessionDeleteCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class SessionAdditionalDirectoriesCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class SessionForkCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class SessionResumeCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class SessionCloseCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class LogoutCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ProvidersCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesDocumentDidOpenCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesDocumentDidCloseCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesDocumentDidSaveCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesDocumentDidFocusCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesRecentFilesCapabilities(BaseModel):
    max_count: Annotated[Optional[int], Field(alias="maxCount", ge=0)] = None
    """
    Maximum number of recent files the agent can use.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("max_count", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NesRelatedSnippetsCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesEditHistoryCapabilities(BaseModel):
    max_count: Annotated[Optional[int], Field(alias="maxCount", ge=0)] = None
    """
    Maximum number of edit history entries the agent can use.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("max_count", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NesUserActionsCapabilities(BaseModel):
    max_count: Annotated[Optional[int], Field(alias="maxCount", ge=0)] = None
    """
    Maximum number of user actions the agent can use.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("max_count", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NesOpenFilesCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesDiagnosticsCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class AuthEnvVar(BaseModel):
    name: str
    """
    The environment variable name (e.g. `"OPENAI_API_KEY"`).
    """
    label: Optional[str] = None
    """
    Human-readable label for this variable, displayed in client UI.
    """
    secret: Optional[bool] = True
    """
    Whether this value is a secret (e.g. API key, token).
    Clients should use a password-style input for secret vars.

    Defaults to `true`.
    """
    optional: Optional[bool] = False
    """
    Whether this variable is optional.

    Defaults to `false`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("optional", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: False)

    @field_validator("label", mode="wrap")
    @classmethod
    def _salvage_on_error_1(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("secret", mode="wrap")
    @classmethod
    def _salvage_on_error_2(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: True)


class AuthMethodEnvVar(BaseModel):
    id: str
    """
    Unique identifier for this authentication method.
    """
    name: str
    """
    Human-readable name of the authentication method.
    """
    description: Optional[str] = None
    """
    Optional description providing more details about this authentication method.
    """
    vars: List[AuthEnvVar]
    """
    The environment variables the client should set.
    """
    link: Optional[str] = None
    """
    Optional link to a page where the user can obtain their credentials.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("description", "link", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("vars", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class AuthMethodTerminal(BaseModel):
    id: str
    """
    Unique identifier for this authentication method.
    """
    name: str
    """
    Human-readable name of the authentication method.
    """
    description: Optional[str] = None
    """
    Optional description providing more details about this authentication method.
    """
    args: Optional[List[str]] = None
    """
    Additional arguments to pass when running the agent binary for terminal auth.
    """
    env: Optional[Dict[str, str]] = None
    """
    Additional environment variables to set when running the agent binary for terminal auth.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("description", "env", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("args", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class AuthMethodAgent(BaseModel):
    id: str
    """
    Unique identifier for this authentication method.
    """
    name: str
    """
    Human-readable name of the authentication method.
    """
    description: Optional[str] = None
    """
    Optional description providing more details about this authentication method.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class Implementation(BaseModel):
    name: str
    """
    Intended for programmatic or logical use, but can be used as a display
    name fallback if title isn’t present.
    """
    title: Optional[str] = None
    """
    Intended for UI and end-user contexts — optimized to be human-readable
    and easily understood.

    If not provided, the name should be used for display.
    """
    version: str
    """
    Version of the implementation. Can be displayed to the user or used
    for debugging or metrics purposes. (e.g. "1.0.0").
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class AuthenticateResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ProviderCurrentConfig(BaseModel):
    api_type: Annotated[
        Union[
            Literal["anthropic"],
            Literal["openai"],
            Literal["azure"],
            Literal["vertex"],
            Literal["bedrock"],
            Dict[str, Any],
        ],
        Field(alias="apiType"),
    ]
    """
    Protocol currently used by this provider.
    """
    base_url: Annotated[str, Field(alias="baseUrl")]
    """
    Base URL currently used by this provider.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class SetProviderResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class DisableProviderResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class LogoutResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class SessionMode(BaseModel):
    id: str
    """
    Stable identifier used to refer to this protocol object in later messages.
    """
    name: str
    """
    Human-readable name shown for this protocol object.
    """
    description: Optional[str] = None
    """
    Optional human-readable details shown with this protocol object.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class SessionConfigSelectOption(BaseModel):
    value: str
    """
    Unique identifier for this option value.
    """
    name: str
    """
    Human-readable label for this option value.
    """
    description: Optional[str] = None
    """
    Optional description for this option value.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class SessionConfigBoolean(BaseModel):
    current_value: Annotated[bool, Field(alias="currentValue")]
    """
    The current value of the boolean option.
    """


class SessionInfo(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    Unique identifier for the session
    """
    cwd: str
    """
    The working directory for this session. Must be an absolute path.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = None
    """
    Additional workspace roots reported for this session. Each path must be absolute.

    When present, this is the complete ordered additional-root list reported
    by the Agent. Omitted and empty values are equivalent: the response
    reports no additional roots.
    """
    title: Optional[str] = None
    """
    Human-readable title for the session
    """
    updated_at: Annotated[Optional[str], Field(alias="updatedAt")] = None
    """
    ISO 8601 timestamp of last activity
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("title", "updated_at", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("additional_directories", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class DeleteSessionResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class CloseSessionResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class SetSessionModeResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class Usage(BaseModel):
    total_tokens: Annotated[int, Field(alias="totalTokens", ge=0)]
    """
    Sum of all token types across session.
    """
    input_tokens: Annotated[int, Field(alias="inputTokens", ge=0)]
    """
    Total input tokens across all turns.
    """
    output_tokens: Annotated[int, Field(alias="outputTokens", ge=0)]
    """
    Total output tokens across all turns.
    """
    thought_tokens: Annotated[Optional[int], Field(alias="thoughtTokens", ge=0)] = None
    """
    Total thought/reasoning tokens
    """
    cached_read_tokens: Annotated[Optional[int], Field(alias="cachedReadTokens", ge=0)] = None
    """
    Total cache read tokens.
    """
    cached_write_tokens: Annotated[Optional[int], Field(alias="cachedWriteTokens", ge=0)] = None
    """
    Total cache write tokens.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("cached_read_tokens", "cached_write_tokens", "thought_tokens", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class StartNesResponse(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for the newly started NES session.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class Position(BaseModel):
    line: Annotated[int, Field(ge=0)]
    """
    Zero-based line number.
    """
    character: Annotated[int, Field(ge=0)]
    """
    Zero-based character offset (encoding-dependent).
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesJumpSuggestion(BaseModel):
    id: str
    """
    Unique identifier for accept/reject tracking.
    """
    uri: str
    """
    The file to navigate to.
    """
    position: Position
    """
    The target position within the file.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesRenameSuggestion(BaseModel):
    id: str
    """
    Unique identifier for accept/reject tracking.
    """
    uri: str
    """
    The file URI containing the symbol.
    """
    position: Position
    """
    The position of the symbol to rename.
    """
    new_name: Annotated[str, Field(alias="newName")]
    """
    The new name for the symbol.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesSearchAndReplaceSuggestion(BaseModel):
    id: str
    """
    Unique identifier for accept/reject tracking.
    """
    uri: str
    """
    The file URI to search within.
    """
    search: str
    """
    The text or pattern to find.
    """
    replace: str
    """
    The replacement text.
    """
    is_regex: Annotated[Optional[bool], Field(alias="isRegex")] = None
    """
    Whether `search` is a regular expression. Defaults to `false`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class CloseNesResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class PlanFile(BaseModel):
    plan_id: Annotated[str, Field(alias="planId")]
    """
    The plan ID to update.
    """
    uri: str
    """
    The URI of the file containing the plan.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class PlanMarkdown(BaseModel):
    plan_id: Annotated[str, Field(alias="planId")]
    """
    The plan ID to update.
    """
    content: str
    """
    Markdown content for the plan.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class PlanRemoved(BaseModel):
    plan_id: Annotated[str, Field(alias="planId")]
    """
    The plan ID to remove.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class UnstructuredCommandInput(BaseModel):
    hint: str
    """
    A hint to display when the input hasn't been provided yet
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class _CurrentModeUpdate(BaseModel):
    current_mode_id: Annotated[str, Field(alias="currentModeId")]
    """
    The ID of the current mode
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class _SessionInfoUpdate(BaseModel):
    title: Optional[str] = None
    """
    Human-readable title for the session. Set to null to clear.
    """
    updated_at: Annotated[Optional[str], Field(alias="updatedAt")] = None
    """
    ISO 8601 timestamp of last activity. Set to null to clear.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("title", "updated_at", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class Cost(BaseModel):
    amount: float
    """
    Total cumulative cost for session.
    """
    currency: str
    """
    ISO 4217 currency code (e.g., "USD", "EUR").
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class _UsageUpdate(BaseModel):
    used: Annotated[int, Field(ge=0)]
    """
    Tokens currently in context.
    """
    size: Annotated[int, Field(ge=0)]
    """
    Total context window size in tokens.
    """
    cost: Optional[Cost] = None
    """
    Cumulative session cost (optional).
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("cost", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class CompleteElicitationNotification(BaseModel):
    elicitation_id: Annotated[str, Field(alias="elicitationId")]
    """
    The ID of the elicitation that completed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class MessageMcpNotification(BaseModel):
    connection_id: Annotated[str, Field(alias="connectionId")]
    """
    The MCP-over-ACP connection this message is sent on.
    """
    method: str
    """
    The inner MCP method name.
    """
    params: Optional[Dict[str, Any]] = None
    """
    Optional inner MCP params.

    If omitted or set to `null`, the inner MCP message has no params.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("params", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class FileSystemCapabilities(BaseModel):
    read_text_file: Annotated[Optional[bool], Field(alias="readTextFile")] = False
    """
    Whether the Client supports `fs/read_text_file` requests.
    """
    write_text_file: Annotated[Optional[bool], Field(alias="writeTextFile")] = False
    """
    Whether the Client supports `fs/write_text_file` requests.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("read_text_file", "write_text_file", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: False)


class BooleanConfigOptionCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class PlanCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class AuthCapabilities(BaseModel):
    terminal: Optional[bool] = False
    """
    Whether the client supports `terminal` authentication methods.

    When `true`, the agent may include `terminal` entries in its authentication methods.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("terminal", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: False)


class ElicitationFormCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ElicitationUrlCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesJumpCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesRenameCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesSearchAndReplaceCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class AuthenticateRequest(BaseModel):
    method_id: Annotated[str, Field(alias="methodId")]
    """
    The ID of the authentication method to use.
    Must be one of the methods advertised in the initialize response.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ListProvidersRequest(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class SetProviderRequest(BaseModel):
    provider_id: Annotated[str, Field(alias="providerId")]
    """
    Provider ID to configure.
    """
    api_type: Annotated[
        Union[
            Literal["anthropic"],
            Literal["openai"],
            Literal["azure"],
            Literal["vertex"],
            Literal["bedrock"],
            Dict[str, Any],
        ],
        Field(alias="apiType"),
    ]
    """
    Protocol type for this provider.
    """
    base_url: Annotated[str, Field(alias="baseUrl")]
    """
    Base URL for requests sent through this provider.
    """
    headers: Optional[Dict[str, str]] = None
    """
    Full headers map for this provider.
    May include authorization, routing, or other integration-specific headers.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class DisableProviderRequest(BaseModel):
    provider_id: Annotated[str, Field(alias="providerId")]
    """
    Provider ID to disable.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class LogoutRequest(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class HttpHeader(BaseModel):
    name: str
    """
    The name of the HTTP header.
    """
    value: str
    """
    The value to set for the HTTP header.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class McpServerHttp(BaseModel):
    name: str
    """
    Human-readable name identifying this MCP server.
    """
    url: str
    """
    URL to the MCP server.
    """
    headers: List[HttpHeader]
    """
    HTTP headers to set when making requests to the MCP server.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class McpServerSse(BaseModel):
    name: str
    """
    Human-readable name identifying this MCP server.
    """
    url: str
    """
    URL to the MCP server.
    """
    headers: List[HttpHeader]
    """
    HTTP headers to set when making requests to the MCP server.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class McpServerAcp(BaseModel):
    name: str
    """
    Human-readable name identifying this MCP server.
    """
    server_id: Annotated[str, Field(alias="serverId")]
    """
    Unique identifier for this MCP server, generated by the component providing it.

    Providers MUST NOT reuse an ID for multiple ACP-transport MCP servers that are visible
    on the same ACP connection.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class McpServerStdio(BaseModel):
    name: str
    """
    Human-readable name identifying this MCP server.
    """
    command: str
    """
    Absolute path to the MCP server executable.
    """
    args: List[str]
    """
    Command-line arguments to pass to the MCP server.
    """
    env: List[EnvVariable]
    """
    Environment variables to set when launching the MCP server.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ListSessionsRequest(BaseModel):
    cwd: Optional[str] = None
    """
    Filter sessions by working directory. Must be an absolute path.
    """
    cursor: Optional[str] = None
    """
    Opaque cursor token from a previous response's nextCursor field for cursor-based pagination
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class DeleteSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to delete.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class CloseSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to close.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class SetSessionModeRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to set the mode for.
    """
    mode_id: Annotated[str, Field(alias="modeId")]
    """
    The ID of the mode to set.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class SetSessionConfigOptionBooleanRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to set the configuration option for.
    """
    config_id: Annotated[str, Field(alias="configId")]
    """
    The ID of the configuration option to set.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """
    value: bool
    """
    The boolean value.
    """
    type: Literal["boolean"]


class SetSessionConfigOptionSelectRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to set the configuration option for.
    """
    config_id: Annotated[str, Field(alias="configId")]
    """
    The ID of the configuration option to set.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """
    value: str
    """
    The value ID.
    """


class WorkspaceFolder(BaseModel):
    uri: str
    """
    The URI of the folder.
    """
    name: str
    """
    The display name of the folder.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesRepository(BaseModel):
    name: str
    """
    The repository name.
    """
    owner: str
    """
    The repository owner.
    """
    remote_url: Annotated[str, Field(alias="remoteUrl")]
    """
    The remote URL of the repository.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesRecentFile(BaseModel):
    uri: str
    """
    The URI of the file.
    """
    language_id: Annotated[str, Field(alias="languageId")]
    """
    The language identifier.
    """
    text: str
    """
    The full text content of the file.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesExcerpt(BaseModel):
    start_line: Annotated[int, Field(alias="startLine", ge=0)]
    """
    The start line of the excerpt (zero-based).
    """
    end_line: Annotated[int, Field(alias="endLine", ge=0)]
    """
    The end line of the excerpt (zero-based).
    """
    text: str
    """
    The text content of the excerpt.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesEditHistoryEntry(BaseModel):
    uri: str
    """
    The URI of the edited file.
    """
    diff: str
    """
    A diff representing the edit.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesUserAction(BaseModel):
    action: str
    """
    The kind of action (e.g., "insertChar", "cursorMovement").
    """
    uri: str
    """
    The URI of the file where the action occurred.
    """
    position: Position
    """
    The position where the action occurred.
    """
    timestamp_ms: Annotated[int, Field(alias="timestampMs", ge=0)]
    """
    Timestamp in milliseconds since epoch.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class CloseNesRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the NES session to close.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class WriteTextFileResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ReadTextFileResponse(BaseModel):
    content: str
    """
    Content payload returned by this response.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class DeniedOutcome(BaseModel):
    outcome: Literal["cancelled"]


class SelectedPermissionOutcome(BaseModel):
    option_id: Annotated[str, Field(alias="optionId")]
    """
    The ID of the option the user selected.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class CreateTerminalResponse(BaseModel):
    terminal_id: Annotated[str, Field(alias="terminalId")]
    """
    The unique identifier for the created terminal.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class TerminalExitStatus(BaseModel):
    exit_code: Annotated[Optional[int], Field(alias="exitCode", ge=0)] = None
    """
    The process exit code (may be null if terminated by signal).
    """
    signal: Optional[str] = None
    """
    The signal that terminated the process (may be null if exited normally).
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("exit_code", "signal", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ReleaseTerminalResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class WaitForTerminalExitResponse(BaseModel):
    exit_code: Annotated[Optional[int], Field(alias="exitCode", ge=0)] = None
    """
    The process exit code (may be null if terminated by signal).
    """
    signal: Optional[str] = None
    """
    The signal that terminated the process (may be null if exited normally).
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("exit_code", "signal", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class KillTerminalResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class DeclineElicitationResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """
    action: Literal["decline"]


class CancelElicitationResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """
    action: Literal["cancel"]


class OtherElicitationResponse(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """
    action: str
    """
    Custom or future elicitation action.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """

    @field_validator("action", mode="before")
    @classmethod
    def _reject_known_action(cls, value: Any) -> Any:
        # Restore the schema's `not` clause dropped for codegen: reject the known
        # variants' discriminator values so a malformed known variant fails instead
        # of silently parsing as this catch-all.
        if value in ("accept", "decline", "cancel"):
            raise ValueError("action value is reserved by a known variant")
        return value


class ElicitationContentValue(RootModel[Union[str, int, float, bool, List[str]]]):
    model_config = ConfigDict(use_attribute_docstrings=True)

    root: Union[str, int, float, bool, List[str]]
    """
    Allowed wire representations for [`ElicitationContentValue`].
    """


class ElicitationAcceptAction(BaseModel):
    content: Optional[Dict[str, Any]] = None
    """
    The user-provided content, if any, as an object matching the requested schema.
    """


class ConnectMcpResponse(BaseModel):
    connection_id: Annotated[str, Field(alias="connectionId")]
    """
    The unique identifier for this MCP-over-ACP connection.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class DisconnectMcpResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class CancelNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to cancel operations for.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class DidOpenDocumentNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    uri: str
    """
    The URI of the opened document.
    """
    language_id: Annotated[str, Field(alias="languageId")]
    """
    The language identifier of the document (e.g., "rust", "python").
    """
    version: int
    """
    The version number of the document.
    """
    text: str
    """
    The full text content of the document.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class DidCloseDocumentNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    uri: str
    """
    The URI of the closed document.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class DidSaveDocumentNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    uri: str
    """
    The URI of the saved document.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class AcceptNesNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    id: str
    """
    The ID of the accepted suggestion.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class CancelRequestNotification(BaseModel):
    request_id: Annotated[Optional[Union[int, str]], Field(alias="requestId")]
    """
    The ID of the request to cancel.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class WriteTextFileRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this request.
    """
    path: str
    """
    Absolute path to the file to write.
    """
    content: str
    """
    The text content to write to the file.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class FileEditToolCallContent(Diff):
    type: Literal["diff"]


class TerminalToolCallContent(Terminal):
    type: Literal["terminal"]


class Annotations(BaseModel):
    audience: Optional[List[str]] = None
    """
    Intended recipients for this content, such as the user or assistant.
    """
    last_modified: Annotated[Optional[str], Field(alias="lastModified")] = None
    """
    Timestamp indicating when the underlying resource was last modified.
    """
    priority: Optional[float] = None
    """
    Relative importance of this content when clients choose what to surface.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("last_modified", "priority", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("audience", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class TextContent(BaseModel):
    annotations: Optional[Annotations] = None
    """
    Optional annotations that help clients decide how to display or route this content.
    """
    text: str
    """
    Text payload carried by this content block.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("annotations", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ImageContent(BaseModel):
    annotations: Optional[Annotations] = None
    """
    Optional annotations that help clients decide how to display or route this content.
    """
    data: str
    """
    Base64-encoded media payload.
    """
    mime_type: Annotated[str, Field(alias="mimeType")]
    """
    MIME type describing the encoded media payload.
    """
    uri: Optional[str] = None
    """
    URI associated with this resource or media payload.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("annotations", "uri", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class AudioContent(BaseModel):
    annotations: Optional[Annotations] = None
    """
    Optional annotations that help clients decide how to display or route this content.
    """
    data: str
    """
    Base64-encoded media payload.
    """
    mime_type: Annotated[str, Field(alias="mimeType")]
    """
    MIME type describing the encoded media payload.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("annotations", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ResourceLink(BaseModel):
    annotations: Optional[Annotations] = None
    """
    Optional annotations that help clients decide how to display or route this content.
    """
    description: Optional[str] = None
    """
    Optional human-readable details shown with this protocol object.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type describing the encoded media payload.
    """
    name: str
    """
    Human-readable name shown for this protocol object.
    """
    size: Optional[int] = None
    """
    Optional size of the linked resource in bytes, if known.
    """
    title: Optional[str] = None
    """
    Optional display title for end-user UI.
    """
    uri: str
    """
    URI associated with this resource or media payload.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("annotations", "description", "mime_type", "size", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class EmbeddedResource(BaseModel):
    annotations: Optional[Annotations] = None
    """
    Optional annotations that help clients decide how to display or route this content.
    """
    resource: Union[TextResourceContents, BlobResourceContents]
    """
    Embedded resource payload, either text or binary data.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("annotations", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class PermissionOption(BaseModel):
    option_id: Annotated[str, Field(alias="optionId")]
    """
    Unique identifier for this permission option.
    """
    name: str
    """
    Human-readable label to display to the user.
    """
    kind: PermissionOptionKind
    """
    Hint about the nature of this permission option.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class CreateTerminalRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this request.
    """
    command: str
    """
    The command to execute.
    """
    args: Optional[List[str]] = None
    """
    Array of command arguments.
    """
    env: Optional[List[EnvVariable]] = None
    """
    Environment variables for the command.
    """
    cwd: Optional[str] = None
    """
    Working directory for the command. Must be an absolute path.
    """
    output_byte_limit: Annotated[Optional[int], Field(alias="outputByteLimit", ge=0)] = None
    """
    Maximum number of output bytes to retain.

    When the limit is exceeded, the Client truncates from the beginning of the output
    to stay within the limit.

    The Client MUST ensure truncation happens at a character boundary to maintain valid
    string output, even if this means the retained output is slightly less than the
    specified limit.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("cwd", "output_byte_limit", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("args", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)

    @field_validator("env", mode="wrap")
    @classmethod
    def _skip_invalid_items_1(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class CreateUrlSessionElicitationRequest(ElicitationSessionScope):
    message: str
    """
    A human-readable message describing what input is needed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """
    mode: Literal["url"]
    elicitation_id: Annotated[str, Field(alias="elicitationId")]
    """
    The unique identifier for this elicitation.
    """
    url: AnyUrl
    """
    The URL to direct the user to.
    """


class CreateUrlRequestElicitationRequest(ElicitationRequestScope):
    message: str
    """
    A human-readable message describing what input is needed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """
    mode: Literal["url"]
    elicitation_id: Annotated[str, Field(alias="elicitationId")]
    """
    The unique identifier for this elicitation.
    """
    url: AnyUrl
    """
    The URL to direct the user to.
    """


class ElicitationStringPropertySchema(StringPropertySchema):
    type: Literal["string"]


class ElicitationNumberPropertySchema(NumberPropertySchema):
    type: Literal["number"]


class ElicitationIntegerPropertySchema(IntegerPropertySchema):
    type: Literal["integer"]


class ElicitationBooleanPropertySchema(BooleanPropertySchema):
    type: Literal["boolean"]


class StringMultiSelectItems(_StringMultiSelectItems):
    type: Literal["string"]


class MultiSelectPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional title for the property.
    """
    description: Optional[str] = None
    """
    Human-readable description.
    """
    min_items: Annotated[Optional[int], Field(alias="minItems", ge=0)] = None
    """
    Minimum number of items to select.
    """
    max_items: Annotated[Optional[int], Field(alias="maxItems", ge=0)] = None
    """
    Maximum number of items to select.
    """
    items: Union[StringMultiSelectItems, OtherMultiSelectItems, TitledMultiSelectItems]
    """
    The items definition describing allowed values.
    """
    default: Optional[List[str]] = None
    """
    Default selected values.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("description", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("default", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ConnectMcpRequest(BaseModel):
    server_id: Annotated[str, Field(alias="serverId")]
    """
    The ACP MCP server ID that was provided by the component declaring the MCP server.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class MessageMcpRequest(BaseModel):
    connection_id: Annotated[str, Field(alias="connectionId")]
    """
    The MCP-over-ACP connection this message is sent on.
    """
    method: str
    """
    The inner MCP method name.
    """
    params: Optional[Dict[str, Any]] = None
    """
    Optional inner MCP params.

    If omitted or set to `null`, the inner MCP message has no params.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class SessionCapabilities(BaseModel):
    list: Optional[SessionListCapabilities] = None
    """
    Whether the agent supports `session/list`.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports listing sessions.
    """
    delete: Optional[SessionDeleteCapabilities] = None
    """
    Whether the agent supports `session/delete`.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports deleting sessions from `session/list`.
    """
    additional_directories: Annotated[
        Optional[SessionAdditionalDirectoriesCapabilities],
        Field(alias="additionalDirectories"),
    ] = None
    """
    Whether the agent supports `additionalDirectories` on supported session lifecycle requests.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports `additionalDirectories` on
    supported session lifecycle requests.

    Agents that also support `session/list` may return
    `SessionInfo.additionalDirectories` to report the complete ordered
    additional-root list associated with a listed session.
    """
    fork: Optional[SessionForkCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the agent supports `session/fork`.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports forking sessions.
    """
    resume: Optional[SessionResumeCapabilities] = None
    """
    Whether the agent supports `session/resume`.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports resuming sessions.
    """
    close: Optional[SessionCloseCapabilities] = None
    """
    Whether the agent supports `session/close`.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports closing sessions.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("additional_directories", "close", "delete", "fork", "list", "resume", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class AgentAuthCapabilities(BaseModel):
    logout: Optional[LogoutCapabilities] = None
    """
    Whether the agent supports the logout method.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports the logout method.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("logout", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NesDocumentDidChangeCapabilities(BaseModel):
    sync_kind: Annotated[str, Field(alias="syncKind")]
    """
    The sync kind the agent wants: `"full"` or `"incremental"`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesContextCapabilities(BaseModel):
    recent_files: Annotated[Optional[NesRecentFilesCapabilities], Field(alias="recentFiles")] = None
    """
    Whether the agent wants recent files context.
    """
    related_snippets: Annotated[Optional[NesRelatedSnippetsCapabilities], Field(alias="relatedSnippets")] = None
    """
    Whether the agent wants related snippets context.
    """
    edit_history: Annotated[Optional[NesEditHistoryCapabilities], Field(alias="editHistory")] = None
    """
    Whether the agent wants edit history context.
    """
    user_actions: Annotated[Optional[NesUserActionsCapabilities], Field(alias="userActions")] = None
    """
    Whether the agent wants user actions context.
    """
    open_files: Annotated[Optional[NesOpenFilesCapabilities], Field(alias="openFiles")] = None
    """
    Whether the agent wants open files context.
    """
    diagnostics: Optional[NesDiagnosticsCapabilities] = None
    """
    Whether the agent wants diagnostics context.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator(
        "diagnostics", "edit_history", "open_files", "recent_files", "related_snippets", "user_actions", mode="wrap"
    )
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class EnvVarAuthMethod(AuthMethodEnvVar):
    type: Literal["env_var"]


class TerminalAuthMethod(AuthMethodTerminal):
    type: Literal["terminal"]


class ProviderInfo(BaseModel):
    provider_id: Annotated[str, Field(alias="providerId")]
    """
    Provider identifier, for example "main" or "openai".
    """
    supported: List[
        Union[
            Literal["anthropic"],
            Literal["openai"],
            Literal["azure"],
            Literal["vertex"],
            Literal["bedrock"],
            Dict[str, Any],
        ]
    ]
    """
    Supported protocol types for this provider.
    """
    required: bool
    """
    Whether this provider is mandatory and cannot be disabled via `providers/disable`.
    If true, clients must not call `providers/disable` for this provider ID.
    """
    current: Optional[ProviderCurrentConfig] = None
    """
    Current effective non-secret routing config.
    Null or omitted means provider is disabled.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("supported", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class SessionModeState(BaseModel):
    current_mode_id: Annotated[str, Field(alias="currentModeId")]
    """
    The current mode the Agent is in.
    """
    available_modes: Annotated[List[SessionMode], Field(alias="availableModes")]
    """
    The set of modes that the Agent can operate in
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("available_modes", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class SessionConfigOptionBoolean(SessionConfigBoolean):
    id: str
    """
    Unique identifier for the configuration option.
    """
    name: str
    """
    Human-readable label for the option.
    """
    description: Optional[str] = None
    """
    Optional description for the Client to display to the user.
    """
    category: Optional[
        Union[
            Literal["mode"],
            Literal["model"],
            Literal["model_config"],
            Literal["thought_level"],
            Dict[str, Any],
        ]
    ] = None
    """
    Optional semantic category for this option (UX only).
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """
    type: Literal["boolean"]

    @field_validator("category", "description", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class SessionConfigSelectGroup(BaseModel):
    group: str
    """
    Unique identifier for this group.
    """
    name: str
    """
    Human-readable label for this group.
    """
    options: List[SessionConfigSelectOption]
    """
    The set of option values in this group.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("options", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ListSessionsResponse(BaseModel):
    sessions: List[SessionInfo]
    """
    Array of session information objects
    """
    next_cursor: Annotated[Optional[str], Field(alias="nextCursor")] = None
    """
    Opaque cursor token. If present, pass this in the next request's cursor parameter
    to fetch the next page. If absent, there are no more results.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("next_cursor", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("sessions", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class PromptResponse(BaseModel):
    stop_reason: Annotated[StopReason, Field(alias="stopReason")]
    """
    Indicates why the agent stopped processing the turn.
    """
    usage: Optional[Usage] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Token usage for this turn (optional).
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("usage", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NesJumpSuggestionVariant(NesJumpSuggestion):
    kind: Literal["jump"]


class NesRenameSuggestionVariant(NesRenameSuggestion):
    kind: Literal["rename"]


class NesSearchAndReplaceSuggestionVariant(NesSearchAndReplaceSuggestion):
    kind: Literal["searchAndReplace"]


class Range(BaseModel):
    start: Position
    """
    The start position (inclusive).
    """
    end: Position
    """
    The end position (exclusive).
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class Error(BaseModel):
    code: Union[
        Literal[-32700],
        Literal[-32600],
        Literal[-32601],
        Literal[-32602],
        Literal[-32603],
        Literal[-32800],
        Literal[-32000],
        Literal[-32002],
        int,
    ]
    """
    A number indicating the error type that occurred.
    This must be an integer as defined in the JSON-RPC specification.
    """
    message: str
    """
    A string providing a short description of the error.
    The message should be limited to a concise single sentence.
    """
    data: Optional[Any] = None
    """
    Optional primitive or structured value that contains additional information about the error.
    This may include debugging information or context-specific details.
    """

    @field_validator("data", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class AgentPlanRemovedUpdate(PlanRemoved):
    session_update: Annotated[Literal["plan_removed"], Field(alias="sessionUpdate")]


class CurrentModeUpdate(_CurrentModeUpdate):
    session_update: Annotated[Literal["current_mode_update"], Field(alias="sessionUpdate")]


class SessionInfoUpdate(_SessionInfoUpdate):
    session_update: Annotated[Literal["session_info_update"], Field(alias="sessionUpdate")]


class UsageUpdate(_UsageUpdate):
    session_update: Annotated[Literal["usage_update"], Field(alias="sessionUpdate")]


class PlanEntry(BaseModel):
    content: str
    """
    Human-readable description of what this task aims to accomplish.
    """
    priority: PlanEntryPriority
    """
    The relative importance of this task.
    Used to indicate which tasks are most critical to the overall goal.
    """
    status: PlanEntryStatus
    """
    Current execution status of this task.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class Plan(BaseModel):
    entries: List[PlanEntry]
    """
    The list of tasks to be accomplished.

    When updating a plan, the agent must send a complete list of all entries
    with their current status. The client replaces the entire plan with each update.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("entries", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class PlanUpdateFile(PlanFile):
    type: Literal["file"]


class PlanUpdateMarkdown(PlanMarkdown):
    type: Literal["markdown"]


class PlanItems(BaseModel):
    plan_id: Annotated[str, Field(alias="planId")]
    """
    The plan ID to update.
    """
    entries: List[PlanEntry]
    """
    The list of tasks to be accomplished.

    When updating an item-based plan, the agent must send a complete list of all entries
    with their current status. The client replaces that plan with each update.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("entries", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class AvailableCommandInput(RootModel[UnstructuredCommandInput]):
    model_config = ConfigDict(use_attribute_docstrings=True)

    root: UnstructuredCommandInput
    """
    The input specification for a command.
    """


class SessionConfigOptionsCapabilities(BaseModel):
    boolean: Optional[BooleanConfigOptionCapabilities] = None
    """
    Whether the client supports boolean session configuration options.

    Optional. Omitted or `null` both mean the client does not advertise support.
    Supplying `{}` means agents may include `type: "boolean"` entries in
    `configOptions`, and the client may send `session/set_config_option`
    requests with `type: "boolean"` and a boolean `value`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("boolean", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ElicitationCapabilities(BaseModel):
    form: Optional[ElicitationFormCapabilities] = None
    """
    Whether the client supports form-based elicitation.

    Optional. Omitted or `null` both mean the client does not advertise support.
    Supplying `{}` means the client supports form-based elicitation.
    """
    url: Optional[ElicitationUrlCapabilities] = None
    """
    Whether the client supports URL-based elicitation.

    Optional. Omitted or `null` both mean the client does not advertise support.
    Supplying `{}` means the client supports URL-based elicitation.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("form", "url", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ClientNesCapabilities(BaseModel):
    jump: Optional[NesJumpCapabilities] = None
    """
    Whether the client supports the `jump` suggestion kind.
    """
    rename: Optional[NesRenameCapabilities] = None
    """
    Whether the client supports the `rename` suggestion kind.
    """
    search_and_replace: Annotated[Optional[NesSearchAndReplaceCapabilities], Field(alias="searchAndReplace")] = None
    """
    Whether the client supports the `searchAndReplace` suggestion kind.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("jump", "rename", "search_and_replace", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class HttpMcpServer(McpServerHttp):
    type: Literal["http"]


class SseMcpServer(McpServerSse):
    type: Literal["sse"]


class AcpMcpServer(McpServerAcp):
    type: Literal["acp"]


class LoadSessionRequest(BaseModel):
    mcp_servers: Annotated[
        List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]],
        Field(alias="mcpServers"),
    ]
    """
    List of MCP servers to connect to for this session.
    """
    cwd: str
    """
    The working directory for this session. Must be an absolute path.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = None
    """
    Additional workspace roots to activate for this session. Each path must be absolute.

    When omitted or empty, no additional roots are activated. When non-empty,
    this is the complete resulting additional-root list for the loaded
    session. It may differ from any previously used or reported list as long as
    the request `cwd` matches the session's `cwd`.
    """
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to load.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("additional_directories", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)

    @field_validator("mcp_servers", mode="wrap")
    @classmethod
    def _skip_invalid_items_1(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ForkSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to fork.
    """
    cwd: str
    """
    The working directory for this session. Must be an absolute path.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = None
    """
    Additional workspace roots to activate for this session. Each path must be absolute.

    When omitted or empty, no additional roots are activated. When non-empty,
    this is the complete resulting additional-root list for the forked
    session.
    """
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]]],
        Field(alias="mcpServers"),
    ] = None
    """
    List of MCP servers to connect to for this session.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("additional_directories", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)

    @field_validator("mcp_servers", mode="wrap")
    @classmethod
    def _skip_invalid_items_1(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ResumeSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to resume.
    """
    cwd: str
    """
    The working directory for this session. Must be an absolute path.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = None
    """
    Additional workspace roots to activate for this session. Each path must be absolute.

    When omitted or empty, no additional roots are activated. When non-empty,
    this is the complete resulting additional-root list for the resumed
    session. It may differ from any previously used or reported list as long as
    the request `cwd` matches the session's `cwd`.
    """
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]]],
        Field(alias="mcpServers"),
    ] = None
    """
    List of MCP servers to connect to for this session.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("additional_directories", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)

    @field_validator("mcp_servers", mode="wrap")
    @classmethod
    def _skip_invalid_items_1(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class StartNesRequest(BaseModel):
    workspace_uri: Annotated[Optional[str], Field(alias="workspaceUri")] = None
    """
    The root URI of the workspace.
    """
    workspace_folders: Annotated[Optional[List[WorkspaceFolder]], Field(alias="workspaceFolders")] = None
    """
    The workspace folders.
    """
    repository: Optional[NesRepository] = None
    """
    Repository metadata, if the workspace is a git repository.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("repository", "workspace_uri", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NesRelatedSnippet(BaseModel):
    uri: str
    """
    The URI of the file containing the snippets.
    """
    excerpts: List[NesExcerpt]
    """
    The code excerpts.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesOpenFile(BaseModel):
    uri: str
    """
    The URI of the file.
    """
    language_id: Annotated[str, Field(alias="languageId")]
    """
    The language identifier.
    """
    visible_range: Annotated[Optional[Range], Field(alias="visibleRange")] = None
    """
    The visible range in the editor, if any.
    """
    last_focused_ms: Annotated[Optional[int], Field(alias="lastFocusedMs", ge=0)] = None
    """
    Timestamp in milliseconds since epoch of when the file was last focused.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("last_focused_ms", "visible_range", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NesDiagnostic(BaseModel):
    uri: str
    """
    The URI of the file containing the diagnostic.
    """
    range: Range
    """
    The range of the diagnostic.
    """
    severity: str
    """
    The severity of the diagnostic.
    """
    message: str
    """
    The diagnostic message.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ClientErrorMessage(BaseModel):
    id: Optional[Union[int, str]]
    """
    The id of the request this response answers.
    """
    error: Error
    """
    Method-specific error data.
    """


class AllowedOutcome(SelectedPermissionOutcome):
    outcome: Literal["selected"]


class TerminalOutputResponse(BaseModel):
    output: str
    """
    The terminal output captured so far.
    """
    truncated: bool
    """
    Whether the output was truncated due to byte limits.
    """
    exit_status: Annotated[Optional[TerminalExitStatus], Field(alias="exitStatus")] = None
    """
    Exit status if the command has completed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("exit_status", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class AcceptElicitationResponse(ElicitationAcceptAction):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """
    action: Literal["accept"]


class TextDocumentContentChangeEvent(BaseModel):
    range: Optional[Range] = None
    """
    The range of the document that changed. If `None`, the entire content is replaced.
    """
    text: str
    """
    The new text for the range, or the full document content if `range` is `None`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class DidFocusDocumentNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    uri: str
    """
    The URI of the focused document.
    """
    version: int
    """
    The version number of the document.
    """
    position: Position
    """
    The current cursor position.
    """
    visible_range: Annotated[Range, Field(alias="visibleRange")]
    """
    The portion of the file currently visible in the editor viewport.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class RejectNesNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    id: str
    """
    The ID of the rejected suggestion.
    """
    reason: Optional[str] = None
    """
    The reason for rejection.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("reason", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class TextContentBlock(TextContent):
    type: Literal["text"]


class ImageContentBlock(ImageContent):
    type: Literal["image"]


class AudioContentBlock(AudioContent):
    type: Literal["audio"]


class ResourceContentBlock(ResourceLink):
    type: Literal["resource_link"]


class EmbeddedResourceContentBlock(EmbeddedResource):
    type: Literal["resource"]


class Content(BaseModel):
    content: Annotated[
        Union[
            TextContentBlock, ImageContentBlock, AudioContentBlock, ResourceContentBlock, EmbeddedResourceContentBlock
        ],
        Field(discriminator="type"),
    ]
    """
    The actual content block.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ElicitationMultiSelectPropertySchema(MultiSelectPropertySchema):
    type: Literal["array"]


class AgentErrorMessage(BaseModel):
    id: Optional[Union[int, str]]
    """
    The id of the request this response answers.
    """
    error: Error
    """
    Method-specific error data.
    """


class NesDocumentEventCapabilities(BaseModel):
    did_open: Annotated[Optional[NesDocumentDidOpenCapabilities], Field(alias="didOpen")] = None
    """
    Whether the agent wants `document/didOpen` events.
    """
    did_change: Annotated[Optional[NesDocumentDidChangeCapabilities], Field(alias="didChange")] = None
    """
    Whether the agent wants `document/didChange` events, and the sync kind.
    """
    did_close: Annotated[Optional[NesDocumentDidCloseCapabilities], Field(alias="didClose")] = None
    """
    Whether the agent wants `document/didClose` events.
    """
    did_save: Annotated[Optional[NesDocumentDidSaveCapabilities], Field(alias="didSave")] = None
    """
    Whether the agent wants `document/didSave` events.
    """
    did_focus: Annotated[Optional[NesDocumentDidFocusCapabilities], Field(alias="didFocus")] = None
    """
    Whether the agent wants `document/didFocus` events.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("did_change", "did_close", "did_focus", "did_open", "did_save", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ListProvidersResponse(BaseModel):
    providers: List[ProviderInfo]
    """
    Configurable providers with current routing info suitable for UI display.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class SessionConfigSelect(BaseModel):
    current_value: Annotated[str, Field(alias="currentValue")]
    """
    The currently selected value.
    """
    options: Union[List[SessionConfigSelectOption], List[SessionConfigSelectGroup]]
    """
    The set of selectable options.
    """


class NesTextEdit(BaseModel):
    range: Range
    """
    The range to replace.
    """
    new_text: Annotated[str, Field(alias="newText")]
    """
    The replacement text.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesEditSuggestion(BaseModel):
    id: str
    """
    Unique identifier for accept/reject tracking.
    """
    uri: str
    """
    The URI of the file to edit.
    """
    edits: List[NesTextEdit]
    """
    The text edits to apply.
    """
    cursor_position: Annotated[Optional[Position], Field(alias="cursorPosition")] = None
    """
    Optional suggested cursor position after applying edits.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("cursor_position", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class AgentPlanUpdate(Plan):
    session_update: Annotated[Literal["plan"], Field(alias="sessionUpdate")]


class ContentChunk(BaseModel):
    content: Annotated[
        Union[
            TextContentBlock, ImageContentBlock, AudioContentBlock, ResourceContentBlock, EmbeddedResourceContentBlock
        ],
        Field(discriminator="type"),
    ]
    """
    A single item of content
    """
    message_id: Annotated[Optional[str], Field(alias="messageId")] = None
    """
    A unique identifier for the message this chunk belongs to.

    All chunks belonging to the same message share the same `messageId`.
    A change in `messageId` indicates a new message has started.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("message_id", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class PlanUpdateItems(PlanItems):
    type: Literal["items"]


class PlanUpdate(BaseModel):
    plan: Annotated[
        Union[PlanUpdateItems, PlanUpdateFile, PlanUpdateMarkdown],
        Field(discriminator="type"),
    ]
    """
    The updated plan content.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class AvailableCommand(BaseModel):
    name: str
    """
    Command name (e.g., `create_plan`, `research_codebase`).
    """
    description: str
    """
    Human-readable description of what the command does.
    """
    input: Optional[AvailableCommandInput] = None
    """
    Input for the command if required
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("input", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class _AvailableCommandsUpdate(BaseModel):
    available_commands: Annotated[List[AvailableCommand], Field(alias="availableCommands")]
    """
    Commands the agent can execute
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("available_commands", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ClientSessionCapabilities(BaseModel):
    config_options: Annotated[Optional[SessionConfigOptionsCapabilities], Field(alias="configOptions")] = None
    """
    Config option capabilities supported by the client.

    Omitted or `null` both mean the client does not advertise support for any
    config option extensions.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("config_options", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NewSessionRequest(BaseModel):
    cwd: str
    """
    The working directory for this session. Must be an absolute path.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = None
    """
    Additional workspace roots for this session. Each path must be absolute.

    These expand the session's filesystem scope without changing `cwd`, which
    remains the base for relative paths. When omitted or empty, no
    additional roots are activated for the new session.
    """
    mcp_servers: Annotated[
        List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]],
        Field(alias="mcpServers"),
    ]
    """
    List of MCP (Model Context Protocol) servers the agent should connect to.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("additional_directories", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)

    @field_validator("mcp_servers", mode="wrap")
    @classmethod
    def _skip_invalid_items_1(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class PromptRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to send this user message to
    """
    prompt: List[
        Annotated[
            Union[
                TextContentBlock,
                ImageContentBlock,
                AudioContentBlock,
                ResourceContentBlock,
                EmbeddedResourceContentBlock,
            ],
            Field(discriminator="type"),
        ]
    ]
    """
    The blocks of content that compose the user's message.

    As a baseline, the Agent MUST support [`ContentBlock::Text`] and [`ContentBlock::ResourceLink`],
    while other variants are optionally enabled via [`PromptCapabilities`].

    The Client MUST adapt its interface according to [`PromptCapabilities`].

    The client MAY include referenced pieces of context as either
    [`ContentBlock::Resource`] or [`ContentBlock::ResourceLink`].

    When available, [`ContentBlock::Resource`] is preferred
    as it avoids extra round-trips and allows the message to include
    pieces of context from sources the agent may not have access to.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class NesSuggestContext(BaseModel):
    recent_files: Annotated[Optional[List[NesRecentFile]], Field(alias="recentFiles")] = None
    """
    Recently accessed files.
    """
    related_snippets: Annotated[Optional[List[NesRelatedSnippet]], Field(alias="relatedSnippets")] = None
    """
    Related code snippets.
    """
    edit_history: Annotated[Optional[List[NesEditHistoryEntry]], Field(alias="editHistory")] = None
    """
    Recent edit history.
    """
    user_actions: Annotated[Optional[List[NesUserAction]], Field(alias="userActions")] = None
    """
    Recent user actions (typing, navigation, etc.).
    """
    open_files: Annotated[Optional[List[NesOpenFile]], Field(alias="openFiles")] = None
    """
    Currently open files in the editor.
    """
    diagnostics: Optional[List[NesDiagnostic]] = None
    """
    Current diagnostics (errors, warnings).
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class RequestPermissionResponse(BaseModel):
    outcome: Annotated[
        Union[DeniedOutcome, AllowedOutcome],
        Field(discriminator="outcome"),
    ]
    """
    The user's decision on the permission request.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class DidChangeDocumentNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    uri: str
    """
    The URI of the changed document.
    """
    version: int
    """
    The new version number of the document.
    """
    content_changes: Annotated[List[TextDocumentContentChangeEvent], Field(alias="contentChanges")]
    """
    The content changes.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("content_changes", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ContentToolCallContent(Content):
    type: Literal["content"]


class ElicitationSchema(BaseModel):
    type: Optional[str] = "object"
    """
    Type discriminator. Always `"object"`.
    """
    title: Optional[str] = None
    """
    Optional title for the schema.
    """
    properties: Annotated[
        Optional[
            Dict[
                str,
                Union[
                    ElicitationStringPropertySchema,
                    ElicitationNumberPropertySchema,
                    ElicitationIntegerPropertySchema,
                    ElicitationBooleanPropertySchema,
                    ElicitationMultiSelectPropertySchema,
                    ElicitationOtherPropertySchema,
                ],
            ]
        ],
        Field(validate_default=True),
    ] = {}
    """
    Property definitions (must be primitive types).
    """
    required: Optional[List[str]] = None
    """
    List of required property names.
    """
    description: Optional[str] = None
    """
    Optional description of what this schema represents.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("type", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: "object")

    @field_validator("description", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_1(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ElicitationFormSessionMode(ElicitationSessionScope):
    requested_schema: Annotated[ElicitationSchema, Field(alias="requestedSchema")]
    """
    A JSON Schema describing the form fields to present to the user.
    """


class ElicitationFormRequestMode(ElicitationRequestScope):
    requested_schema: Annotated[ElicitationSchema, Field(alias="requestedSchema")]
    """
    A JSON Schema describing the form fields to present to the user.
    """


class ElicitationFormMode(RootModel[Union[ElicitationFormSessionMode, ElicitationFormRequestMode]]):
    model_config = ConfigDict(use_attribute_docstrings=True)

    root: Union[ElicitationFormSessionMode, ElicitationFormRequestMode]
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Form-based elicitation mode where the client renders a form from the provided schema.
    """


class NesEventCapabilities(BaseModel):
    document: Optional[NesDocumentEventCapabilities] = None
    """
    Document event capabilities.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("document", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class SessionConfigOptionSelect(SessionConfigSelect):
    id: str
    """
    Unique identifier for the configuration option.
    """
    name: str
    """
    Human-readable label for the option.
    """
    description: Optional[str] = None
    """
    Optional description for the Client to display to the user.
    """
    category: Optional[
        Union[
            Literal["mode"],
            Literal["model"],
            Literal["model_config"],
            Literal["thought_level"],
            Dict[str, Any],
        ]
    ] = None
    """
    Optional semantic category for this option (UX only).
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """
    type: Literal["select"]

    @field_validator("category", "description", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class LoadSessionResponse(BaseModel):
    modes: Optional[SessionModeState] = None
    """
    Initial mode state if supported by the Agent

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    """
    config_options: Annotated[
        Optional[
            List[
                Annotated[
                    Union[SessionConfigOptionSelect, SessionConfigOptionBoolean],
                    Field(discriminator="type"),
                ]
            ]
        ],
        Field(alias="configOptions"),
    ] = None
    """
    Initial session configuration options if supported by the Agent.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("modes", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("config_options", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ForkSessionResponse(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    Unique identifier for the newly created forked session.
    """
    modes: Optional[SessionModeState] = None
    """
    Initial mode state if supported by the Agent

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    """
    config_options: Annotated[
        Optional[
            List[
                Annotated[
                    Union[SessionConfigOptionSelect, SessionConfigOptionBoolean],
                    Field(discriminator="type"),
                ]
            ]
        ],
        Field(alias="configOptions"),
    ] = None
    """
    Initial session configuration options if supported by the Agent.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("modes", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("config_options", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ResumeSessionResponse(BaseModel):
    modes: Optional[SessionModeState] = None
    """
    Initial mode state if supported by the Agent

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    """
    config_options: Annotated[
        Optional[
            List[
                Annotated[
                    Union[SessionConfigOptionSelect, SessionConfigOptionBoolean],
                    Field(discriminator="type"),
                ]
            ]
        ],
        Field(alias="configOptions"),
    ] = None
    """
    Initial session configuration options if supported by the Agent.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("modes", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("config_options", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class SetSessionConfigOptionResponse(BaseModel):
    config_options: Annotated[
        List[
            Annotated[
                Union[SessionConfigOptionSelect, SessionConfigOptionBoolean],
                Field(discriminator="type"),
            ]
        ],
        Field(alias="configOptions"),
    ]
    """
    The full set of configuration options and their current values.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("config_options", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class NesEditSuggestionVariant(NesEditSuggestion):
    kind: Literal["edit"]


class UserMessageChunk(ContentChunk):
    session_update: Annotated[Literal["user_message_chunk"], Field(alias="sessionUpdate")]


class AgentMessageChunk(ContentChunk):
    session_update: Annotated[Literal["agent_message_chunk"], Field(alias="sessionUpdate")]


class AgentThoughtChunk(ContentChunk):
    session_update: Annotated[Literal["agent_thought_chunk"], Field(alias="sessionUpdate")]


class AgentPlanContentUpdate(PlanUpdate):
    session_update: Annotated[Literal["plan_update"], Field(alias="sessionUpdate")]


class AvailableCommandsUpdate(_AvailableCommandsUpdate):
    session_update: Annotated[Literal["available_commands_update"], Field(alias="sessionUpdate")]


class ToolCall(BaseModel):
    tool_call_id: Annotated[str, Field(alias="toolCallId")]
    """
    Unique identifier for this tool call within the session.
    """
    title: str
    """
    Human-readable title describing what the tool is doing.
    """
    kind: Optional[ToolKind] = None
    """
    The category of tool being invoked.
    Helps clients choose appropriate icons and UI treatment.
    """
    status: Optional[ToolCallStatus] = None
    """
    Current execution status of the tool call.
    """
    content: Optional[
        List[
            Annotated[
                Union[ContentToolCallContent, FileEditToolCallContent, TerminalToolCallContent],
                Field(discriminator="type"),
            ]
        ]
    ] = None
    """
    Content produced by the tool call.
    """
    locations: Optional[List[ToolCallLocation]] = None
    """
    File locations affected by this tool call.
    Enables "follow-along" features in clients.
    """
    raw_input: Annotated[Optional[Any], Field(alias="rawInput")] = None
    """
    Raw input parameters sent to the tool.
    """
    raw_output: Annotated[Optional[Any], Field(alias="rawOutput")] = None
    """
    Raw output returned by the tool.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("kind", "raw_input", "raw_output", "status", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("content", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)

    @field_validator("locations", mode="wrap")
    @classmethod
    def _skip_invalid_items_1(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class _ConfigOptionUpdate(BaseModel):
    config_options: Annotated[
        List[
            Annotated[
                Union[SessionConfigOptionSelect, SessionConfigOptionBoolean],
                Field(discriminator="type"),
            ]
        ],
        Field(alias="configOptions"),
    ]
    """
    The full set of configuration options and their current values.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("config_options", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ClientCapabilities(BaseModel):
    fs: Annotated[Optional[FileSystemCapabilities], Field(validate_default=True)] = FileSystemCapabilities()
    """
    File system capabilities supported by the client.
    Determines which file operations the agent can request.
    """
    terminal: Optional[bool] = False
    """
    Whether the Client support all `terminal/*` methods.
    """
    session: Optional[ClientSessionCapabilities] = None
    """
    Session-related capabilities supported by the client.

    Optional. Omitted or `null` both mean the client does not advertise any
    session-related extensions.
    """
    plan: Optional[PlanCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the client supports `plan_update` and `plan_removed` session updates.

    Optional. Omitted or `null` both mean the client does not advertise support.
    Supplying `{}` means the client can receive both update types.
    """
    auth: Annotated[Optional[AuthCapabilities], Field(validate_default=True)] = {"terminal": False}
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Authentication capabilities supported by the client.
    Determines which authentication method types the agent may include
    in its `InitializeResponse`.
    """
    elicitation: Optional[ElicitationCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Elicitation capabilities supported by the client.
    Determines which elicitation modes the agent may use.

    Optional. Omitted or `null` both mean the client does not advertise
    elicitation support.
    """
    nes: Optional[ClientNesCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    NES (Next Edit Suggestions) capabilities supported by the client.

    Optional. Omitted or `null` both mean the client does not advertise any
    NES suggestion-kind extensions.
    """
    position_encodings: Annotated[Optional[List[str]], Field(alias="positionEncodings")] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    The position encodings supported by the client, in order of preference.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("terminal", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: False)

    @field_validator("elicitation", "nes", "plan", "session", mode="wrap")
    @classmethod
    def _salvage_on_error_1(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("fs", mode="wrap")
    @classmethod
    def _salvage_on_error_2(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: {"readTextFile": False, "writeTextFile": False})

    @field_validator("auth", mode="wrap")
    @classmethod
    def _salvage_on_error_3(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: {"terminal": False})

    @field_validator("position_encodings", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class SuggestNesRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this request.
    """
    uri: str
    """
    The URI of the document to suggest for.
    """
    version: int
    """
    The version number of the document.
    """
    position: Position
    """
    The current cursor position.
    """
    selection: Optional[Range] = None
    """
    The current text selection range, if any.
    """
    trigger_kind: Annotated[str, Field(alias="triggerKind")]
    """
    What triggered this suggestion request.
    """
    context: Optional[NesSuggestContext] = None
    """
    Context for the suggestion, included based on agent capabilities.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ClientResponseMessage(BaseModel):
    id: Optional[Union[int, str]]
    """
    The id of the request this response answers.
    """
    result: Union[
        WriteTextFileResponse,
        ReadTextFileResponse,
        RequestPermissionResponse,
        CreateTerminalResponse,
        TerminalOutputResponse,
        ReleaseTerminalResponse,
        WaitForTerminalExitResponse,
        KillTerminalResponse,
        ConnectMcpResponse,
        DisconnectMcpResponse,
        Union[
            AcceptElicitationResponse,
            DeclineElicitationResponse,
            CancelElicitationResponse,
            OtherElicitationResponse,
        ],
        Any,
    ]
    """
    Method-specific response data.
    """


class ClientResponse(RootModel[Union[ClientResponseMessage, ClientErrorMessage]]):
    model_config = ConfigDict(use_attribute_docstrings=True)

    root: Union[ClientResponseMessage, ClientErrorMessage]
    """
    A JSON-RPC response object.
    """


class ClientNotification(BaseModel):
    method: str
    """
    The notification method name.
    """
    params: Optional[
        Union[
            CancelNotification,
            DidOpenDocumentNotification,
            DidChangeDocumentNotification,
            DidCloseDocumentNotification,
            DidSaveDocumentNotification,
            DidFocusDocumentNotification,
            AcceptNesNotification,
            RejectNesNotification,
            MessageMcpNotification,
            Any,
        ]
    ] = None
    """
    Method-specific notification parameters.
    """


class ToolCallUpdate(BaseModel):
    tool_call_id: Annotated[str, Field(alias="toolCallId")]
    """
    The ID of the tool call being updated.
    """
    kind: Optional[ToolKind] = None
    """
    Update the tool kind.
    """
    status: Optional[ToolCallStatus] = None
    """
    Update the execution status.
    """
    title: Optional[str] = None
    """
    Update the human-readable title.
    """
    content: Optional[
        List[
            Annotated[
                Union[ContentToolCallContent, FileEditToolCallContent, TerminalToolCallContent],
                Field(discriminator="type"),
            ]
        ]
    ] = None
    """
    Replace the content collection.
    """
    locations: Optional[List[ToolCallLocation]] = None
    """
    Replace the locations collection.
    """
    raw_input: Annotated[Optional[Any], Field(alias="rawInput")] = None
    """
    Update the raw input.
    """
    raw_output: Annotated[Optional[Any], Field(alias="rawOutput")] = None
    """
    Update the raw output.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("kind", "raw_input", "raw_output", "status", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("content", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)

    @field_validator("locations", mode="wrap")
    @classmethod
    def _skip_invalid_items_1(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class CreateFormSessionElicitationRequest(ElicitationSessionScope):
    message: str
    """
    A human-readable message describing what input is needed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """
    mode: Literal["form"]
    requested_schema: Annotated[ElicitationSchema, Field(alias="requestedSchema")]
    """
    A JSON Schema describing the form fields to present to the user.
    """


class CreateFormRequestElicitationRequest(ElicitationRequestScope):
    message: str
    """
    A human-readable message describing what input is needed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """
    mode: Literal["form"]
    requested_schema: Annotated[ElicitationSchema, Field(alias="requestedSchema")]
    """
    A JSON Schema describing the form fields to present to the user.
    """


ElicitationMode = Union[
    ElicitationFormSessionMode,
    ElicitationFormRequestMode,
    ElicitationUrlSessionMode,
    ElicitationUrlRequestMode,
]
CreateFormElicitationRequest = Union[
    CreateFormSessionElicitationRequest,
    CreateFormRequestElicitationRequest,
]
CreateUrlElicitationRequest = Union[
    CreateUrlSessionElicitationRequest,
    CreateUrlRequestElicitationRequest,
]
CreateElicitationRequest = Union[
    CreateFormElicitationRequest,
    CreateUrlElicitationRequest,
    CreateOtherElicitationRequest,
]
CreateElicitationResponse = Union[
    AcceptElicitationResponse,
    DeclineElicitationResponse,
    CancelElicitationResponse,
    OtherElicitationResponse,
]


class NesCapabilities(BaseModel):
    events: Optional[NesEventCapabilities] = None
    """
    Events the agent wants to receive.
    """
    context: Optional[NesContextCapabilities] = None
    """
    Context the agent wants attached to each suggestion request.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("context", "events", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NewSessionResponse(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    Unique identifier for the created session.

    Used in all subsequent requests for this conversation.
    """
    modes: Optional[SessionModeState] = None
    """
    Initial mode state if supported by the Agent

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    """
    config_options: Annotated[
        Optional[
            List[
                Annotated[
                    Union[SessionConfigOptionSelect, SessionConfigOptionBoolean],
                    Field(discriminator="type"),
                ]
            ]
        ],
        Field(alias="configOptions"),
    ] = None
    """
    Initial session configuration options if supported by the Agent.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("modes", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("config_options", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class SuggestNesResponse(BaseModel):
    suggestions: List[
        Annotated[
            Union[
                NesEditSuggestionVariant,
                NesJumpSuggestionVariant,
                NesRenameSuggestionVariant,
                NesSearchAndReplaceSuggestionVariant,
            ],
            Field(discriminator="kind"),
        ]
    ]
    """
    The list of suggestions.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ToolCallStart(ToolCall):
    session_update: Annotated[Literal["tool_call"], Field(alias="sessionUpdate")]


class ToolCallProgress(ToolCallUpdate):
    session_update: Annotated[Literal["tool_call_update"], Field(alias="sessionUpdate")]


class ConfigOptionUpdate(_ConfigOptionUpdate):
    session_update: Annotated[Literal["config_option_update"], Field(alias="sessionUpdate")]


class InitializeRequest(BaseModel):
    protocol_version: Annotated[int, Field(alias="protocolVersion", ge=0, le=65535)]
    """
    The latest protocol version supported by the client.
    """
    client_capabilities: Annotated[
        Optional[ClientCapabilities],
        Field(alias="clientCapabilities", validate_default=True),
    ] = ClientCapabilities()
    """
    Capabilities supported by the client.
    """
    client_info: Annotated[Optional[Implementation], Field(alias="clientInfo")] = None
    """
    Information about the Client name and version sent to the Agent.

    Note: in future versions of the protocol, this will be required.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("protocol_version", mode="before")
    @classmethod
    def _coerce_protocol_version(cls, value: Any) -> int:
        # Some clients (e.g. Zed) send a date string like "2024-11-05" instead
        # of an integer. The Rust SDK treats legacy strings as version 0; this
        # SDK maps unparsable values to 1 so the connection is not rejected.
        # See: https://github.com/agentclientprotocol/rust-sdk/blob/main/crates/agent-client-protocol-schema/src/version.rs
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return 1

    @field_validator("client_info", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("client_capabilities", mode="wrap")
    @classmethod
    def _salvage_on_error_1(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(
            value,
            handler,
            lambda: {
                "fs": {"readTextFile": False, "writeTextFile": False},
                "terminal": False,
                "auth": {"terminal": False},
            },
        )


class RequestPermissionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this request.
    """
    tool_call: Annotated[ToolCallUpdate, Field(alias="toolCall")]
    """
    Details about the tool call requiring permission.
    """
    options: List[PermissionOption]
    """
    Available permission options for the user to choose from.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class AgentCapabilities(BaseModel):
    load_session: Annotated[Optional[bool], Field(alias="loadSession")] = False
    """
    Whether the agent supports `session/load`.
    """
    prompt_capabilities: Annotated[
        Optional[PromptCapabilities],
        Field(alias="promptCapabilities", validate_default=True),
    ] = PromptCapabilities()
    """
    Prompt capabilities supported by the agent.
    """
    mcp_capabilities: Annotated[Optional[McpCapabilities], Field(alias="mcpCapabilities", validate_default=True)] = (
        McpCapabilities()
    )
    """
    MCP capabilities supported by the agent.
    """
    session_capabilities: Annotated[
        Optional[SessionCapabilities],
        Field(alias="sessionCapabilities", validate_default=True),
    ] = SessionCapabilities()
    """
    Session lifecycle and prompt capabilities advertised by the agent.
    """
    auth: Annotated[Optional[AgentAuthCapabilities], Field(validate_default=True)] = {}
    """
    Authentication-related capabilities supported by the agent.
    """
    providers: Optional[ProvidersCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Provider configuration capabilities supported by the agent.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports provider configuration methods.
    """
    nes: Optional[NesCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    NES (Next Edit Suggestions) capabilities supported by the agent.

    Optional. Omitted or `null` both mean the agent does not advertise support
    for NES methods.
    """
    position_encoding: Annotated[Optional[str], Field(alias="positionEncoding")] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    The position encoding selected by the agent from the client's supported encodings.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("load_session", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: False)

    @field_validator("nes", "position_encoding", "providers", mode="wrap")
    @classmethod
    def _salvage_on_error_1(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("mcp_capabilities", mode="wrap")
    @classmethod
    def _salvage_on_error_2(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: {"http": False, "sse": False, "acp": False})

    @field_validator("prompt_capabilities", mode="wrap")
    @classmethod
    def _salvage_on_error_3(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: {"image": False, "audio": False, "embeddedContext": False})

    @field_validator("auth", "session_capabilities", mode="wrap")
    @classmethod
    def _salvage_on_error_4(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: {})


class SessionNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session this update pertains to.
    """
    update: Annotated[
        Union[
            UserMessageChunk,
            AgentMessageChunk,
            AgentThoughtChunk,
            ToolCallStart,
            ToolCallProgress,
            AgentPlanUpdate,
            AgentPlanContentUpdate,
            AgentPlanRemovedUpdate,
            AvailableCommandsUpdate,
            CurrentModeUpdate,
            ConfigOptionUpdate,
            SessionInfoUpdate,
            UsageUpdate,
        ],
        Field(discriminator="session_update"),
    ]
    """
    The actual update content.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """


class ClientRequest(BaseModel):
    id: Optional[Union[int, str]]
    """
    The request id used to correlate the matching response.
    """
    method: str
    """
    The method name to invoke.
    """
    params: Optional[
        Union[
            InitializeRequest,
            AuthenticateRequest,
            ListProvidersRequest,
            SetProviderRequest,
            DisableProviderRequest,
            LogoutRequest,
            NewSessionRequest,
            LoadSessionRequest,
            ListSessionsRequest,
            DeleteSessionRequest,
            ForkSessionRequest,
            ResumeSessionRequest,
            CloseSessionRequest,
            SetSessionModeRequest,
            PromptRequest,
            StartNesRequest,
            SuggestNesRequest,
            CloseNesRequest,
            MessageMcpRequest,
            Union[SetSessionConfigOptionBooleanRequest, SetSessionConfigOptionSelectRequest],
            Any,
        ]
    ] = None
    """
    Method-specific request parameters.
    """


class AgentRequest(BaseModel):
    id: Optional[Union[int, str]]
    """
    The request id used to correlate the matching response.
    """
    method: str
    """
    The method name to invoke.
    """
    params: Optional[
        Union[
            WriteTextFileRequest,
            ReadTextFileRequest,
            RequestPermissionRequest,
            CreateTerminalRequest,
            TerminalOutputRequest,
            ReleaseTerminalRequest,
            WaitForTerminalExitRequest,
            KillTerminalRequest,
            ConnectMcpRequest,
            MessageMcpRequest,
            DisconnectMcpRequest,
            Union[
                CreateFormSessionElicitationRequest,
                CreateFormRequestElicitationRequest,
                CreateUrlSessionElicitationRequest,
                CreateUrlRequestElicitationRequest,
                CreateOtherElicitationRequest,
            ],
            Any,
        ]
    ] = None
    """
    Method-specific request parameters.
    """


class InitializeResponse(BaseModel):
    protocol_version: Annotated[int, Field(alias="protocolVersion", ge=0, le=65535)]
    """
    The protocol version the client specified if supported by the agent,
    or the latest protocol version supported by the agent.

    The client should disconnect, if it doesn't support this version.
    """
    agent_capabilities: Annotated[
        Optional[AgentCapabilities],
        Field(alias="agentCapabilities", validate_default=True),
    ] = AgentCapabilities()
    """
    Capabilities supported by the agent.
    """
    auth_methods: Annotated[
        Optional[List[Union[EnvVarAuthMethod, TerminalAuthMethod, AuthMethodAgent]]],
        Field(alias="authMethods", validate_default=True),
    ] = []
    """
    Authentication methods supported by the agent.
    """
    agent_info: Annotated[Optional[Implementation], Field(alias="agentInfo")] = None
    """
    Information about the Agent name and version sent to the Client.

    Note: in future versions of the protocol, this will be required.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    """

    @field_validator("agent_info", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("agent_capabilities", mode="wrap")
    @classmethod
    def _salvage_on_error_1(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(
            value,
            handler,
            lambda: {
                "loadSession": False,
                "promptCapabilities": {"image": False, "audio": False, "embeddedContext": False},
                "mcpCapabilities": {"http": False, "sse": False, "acp": False},
                "sessionCapabilities": {},
                "auth": {},
            },
        )

    @field_validator("auth_methods", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class AgentNotification(BaseModel):
    method: str
    """
    The notification method name.
    """
    params: Optional[
        Union[
            SessionNotification,
            CompleteElicitationNotification,
            MessageMcpNotification,
            Any,
        ]
    ] = None
    """
    Method-specific notification parameters.
    """


class AgentResponseMessage(BaseModel):
    id: Optional[Union[int, str]]
    """
    The id of the request this response answers.
    """
    result: Union[
        InitializeResponse,
        AuthenticateResponse,
        ListProvidersResponse,
        SetProviderResponse,
        DisableProviderResponse,
        LogoutResponse,
        NewSessionResponse,
        LoadSessionResponse,
        ListSessionsResponse,
        DeleteSessionResponse,
        ForkSessionResponse,
        ResumeSessionResponse,
        CloseSessionResponse,
        SetSessionModeResponse,
        SetSessionConfigOptionResponse,
        PromptResponse,
        StartNesResponse,
        SuggestNesResponse,
        CloseNesResponse,
        Any,
    ]
    """
    Method-specific response data.
    """


class AgentResponse(RootModel[Union[AgentResponseMessage, AgentErrorMessage]]):
    model_config = ConfigDict(use_attribute_docstrings=True)

    root: Union[AgentResponseMessage, AgentErrorMessage]
    """
    A JSON-RPC response object.
    """
