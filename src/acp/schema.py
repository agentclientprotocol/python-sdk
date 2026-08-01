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
    model_config = ConfigDict(populate_by_name=True)

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
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    path: Annotated[str, Field(description="Absolute path to the file to read.")]
    line: Annotated[
        Optional[int],
        Field(description="Line number to start reading from (1-based).", ge=0),
    ] = None
    limit: Annotated[Optional[int], Field(description="Maximum number of lines to read.", ge=0)] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("limit", "line", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class TextResourceContents(BaseModel):
    mime_type: Annotated[
        Optional[str],
        Field(
            alias="mimeType",
            description="MIME type describing the encoded media payload.",
        ),
    ] = None
    text: Annotated[str, Field(description="Text payload carried by this content block.")]
    uri: Annotated[str, Field(description="URI associated with this resource or media payload.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("mime_type", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class BlobResourceContents(BaseModel):
    blob: Annotated[str, Field(description="Base64-encoded bytes for a binary resource payload.")]
    mime_type: Annotated[
        Optional[str],
        Field(
            alias="mimeType",
            description="MIME type describing the encoded media payload.",
        ),
    ] = None
    uri: Annotated[str, Field(description="URI associated with this resource or media payload.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("mime_type", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class Diff(BaseModel):
    path: Annotated[str, Field(description="The absolute file path being modified.")]
    old_text: Annotated[
        Optional[str],
        Field(alias="oldText", description="The original content (None for new files)."),
    ] = None
    new_text: Annotated[str, Field(alias="newText", description="The new content after modification.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("old_text", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class Terminal(BaseModel):
    terminal_id: Annotated[
        str,
        Field(
            alias="terminalId",
            description="Identifier of the terminal instance to embed in the content stream.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ToolCallLocation(BaseModel):
    path: Annotated[str, Field(description="The absolute file path being accessed or modified.")]
    line: Annotated[Optional[int], Field(description="Optional line number within the file.", ge=0)] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("line", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class EnvVariable(BaseModel):
    name: Annotated[str, Field(description="The name of the environment variable.")]
    value: Annotated[str, Field(description="The value to set for the environment variable.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class TerminalOutputRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    terminal_id: Annotated[
        str,
        Field(alias="terminalId", description="The ID of the terminal to get output from."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ReleaseTerminalRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    terminal_id: Annotated[str, Field(alias="terminalId", description="The ID of the terminal to release.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class WaitForTerminalExitRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    terminal_id: Annotated[
        str,
        Field(alias="terminalId", description="The ID of the terminal to wait for."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class KillTerminalRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    terminal_id: Annotated[str, Field(alias="terminalId", description="The ID of the terminal to kill.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CreateOtherElicitationRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    message: Annotated[
        str,
        Field(description="A human-readable message describing what input is needed."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    mode: Annotated[
        str,
        Field(
            description="Custom or future elicitation mode.\n\nValues beginning with `_` are reserved for implementation-specific\nextensions. Unknown values that do not begin with `_` are reserved for\nfuture ACP variants."
        ),
    ]

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
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session this elicitation is tied to."),
    ]
    tool_call_id: Annotated[
        Optional[str],
        Field(alias="toolCallId", description="Optional tool call within the session."),
    ] = None

    @field_validator("tool_call_id", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ElicitationRequestScope(BaseModel):
    request_id: Annotated[
        Optional[Union[int, str]],
        Field(alias="requestId", description="The request this elicitation is tied to."),
    ]


class ElicitationOtherPropertySchema(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: Annotated[
        str,
        Field(
            description="Custom or future elicitation property schema type.\n\nValues beginning with `_` are reserved for implementation-specific\nextensions. Unknown values that do not begin with `_` are reserved for\nfuture ACP variants."
        ),
    ]

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
    const: Annotated[str, Field(description="The constant value for this option.")]
    title: Annotated[str, Field(description="Human-readable title for this option.")]
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("description", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class StringPropertySchema(BaseModel):
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    min_length: Annotated[
        Optional[int],
        Field(alias="minLength", description="Minimum string length.", ge=0),
    ] = None
    max_length: Annotated[
        Optional[int],
        Field(alias="maxLength", description="Maximum string length.", ge=0),
    ] = None
    pattern: Annotated[Optional[str], Field(description="Pattern the string must match.")] = None
    format: Annotated[Optional[str], Field(description="String format.")] = None
    default: Annotated[Optional[str], Field(description="Default value.")] = None
    enum: Annotated[
        Optional[List[str]],
        Field(description="Enum values for untitled single-select enums."),
    ] = None
    one_of: Annotated[
        Optional[List[EnumOption]],
        Field(
            alias="oneOf",
            description="Titled enum options for titled single-select enums.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NumberPropertySchema(BaseModel):
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    minimum: Annotated[Optional[float], Field(description="Minimum value (inclusive).")] = None
    maximum: Annotated[Optional[float], Field(description="Maximum value (inclusive).")] = None
    default: Annotated[Optional[float], Field(description="Default value.")] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class IntegerPropertySchema(BaseModel):
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    minimum: Annotated[Optional[int], Field(description="Minimum value (inclusive).")] = None
    maximum: Annotated[Optional[int], Field(description="Maximum value (inclusive).")] = None
    default: Annotated[Optional[int], Field(description="Default value.")] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class BooleanPropertySchema(BaseModel):
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    default: Annotated[Optional[bool], Field(description="Default value.")] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class OtherMultiSelectItems(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: Annotated[
        str,
        Field(
            description="Custom or future multi-select item type.\n\nValues beginning with `_` are reserved for implementation-specific\nextensions. Unknown values that do not begin with `_` are reserved for\nfuture ACP variants."
        ),
    ]

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
    enum: Annotated[List[str], Field(description="Allowed enum values.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class TitledMultiSelectItems(BaseModel):
    any_of: Annotated[List[EnumOption], Field(alias="anyOf", description="Titled enum options.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ElicitationUrlSessionMode(ElicitationSessionScope):
    elicitation_id: Annotated[
        str,
        Field(
            alias="elicitationId",
            description="The unique identifier for this elicitation.",
        ),
    ]
    url: Annotated[AnyUrl, Field(description="The URL to direct the user to.")]


class ElicitationUrlRequestMode(ElicitationRequestScope):
    elicitation_id: Annotated[
        str,
        Field(
            alias="elicitationId",
            description="The unique identifier for this elicitation.",
        ),
    ]
    url: Annotated[AnyUrl, Field(description="The URL to direct the user to.")]


class ElicitationUrlMode(RootModel[Union[ElicitationUrlSessionMode, ElicitationUrlRequestMode]]):
    root: Annotated[
        Union[ElicitationUrlSessionMode, ElicitationUrlRequestMode],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nURL-based elicitation mode where the client directs the user to a URL."
        ),
    ]


class DisconnectMcpRequest(BaseModel):
    connection_id: Annotated[
        str,
        Field(alias="connectionId", description="The MCP-over-ACP connection to close."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class PromptCapabilities(BaseModel):
    image: Annotated[Optional[bool], Field(description="Agent supports [`ContentBlock::Image`].")] = False
    audio: Annotated[Optional[bool], Field(description="Agent supports [`ContentBlock::Audio`].")] = False
    embedded_context: Annotated[
        Optional[bool],
        Field(
            alias="embeddedContext",
            description="Agent supports embedded context in `session/prompt` requests.\n\nWhen enabled, the Client is allowed to include [`ContentBlock::Resource`]\nin prompt requests for pieces of context that are referenced in the message.",
        ),
    ] = False
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("audio", "embedded_context", "image", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: False)


class McpCapabilities(BaseModel):
    http: Annotated[Optional[bool], Field(description="Agent supports [`McpServer::Http`].")] = False
    sse: Annotated[Optional[bool], Field(description="Agent supports [`McpServer::Sse`].")] = False
    acp: Annotated[
        Optional[bool],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nAgent supports [`McpServer::Acp`]."
        ),
    ] = False
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("acp", "http", "sse", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: False)


class SessionListCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionDeleteCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionAdditionalDirectoriesCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionForkCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionResumeCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionCloseCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class LogoutCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ProvidersCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesDocumentDidOpenCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesDocumentDidCloseCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesDocumentDidSaveCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesDocumentDidFocusCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesRecentFilesCapabilities(BaseModel):
    max_count: Annotated[
        Optional[int],
        Field(
            alias="maxCount",
            description="Maximum number of recent files the agent can use.",
            ge=0,
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("max_count", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NesRelatedSnippetsCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesEditHistoryCapabilities(BaseModel):
    max_count: Annotated[
        Optional[int],
        Field(
            alias="maxCount",
            description="Maximum number of edit history entries the agent can use.",
            ge=0,
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("max_count", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NesUserActionsCapabilities(BaseModel):
    max_count: Annotated[
        Optional[int],
        Field(
            alias="maxCount",
            description="Maximum number of user actions the agent can use.",
            ge=0,
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("max_count", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NesOpenFilesCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesDiagnosticsCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class AuthEnvVar(BaseModel):
    name: Annotated[
        str,
        Field(description='The environment variable name (e.g. `"OPENAI_API_KEY"`).'),
    ]
    label: Annotated[
        Optional[str],
        Field(description="Human-readable label for this variable, displayed in client UI."),
    ] = None
    secret: Annotated[
        Optional[bool],
        Field(
            description="Whether this value is a secret (e.g. API key, token).\nClients should use a password-style input for secret vars.\n\nDefaults to `true`."
        ),
    ] = True
    optional: Annotated[
        Optional[bool],
        Field(description="Whether this variable is optional.\n\nDefaults to `false`."),
    ] = False
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
    id: Annotated[str, Field(description="Unique identifier for this authentication method.")]
    name: Annotated[str, Field(description="Human-readable name of the authentication method.")]
    description: Annotated[
        Optional[str],
        Field(description="Optional description providing more details about this authentication method."),
    ] = None
    vars: Annotated[
        List[AuthEnvVar],
        Field(description="The environment variables the client should set."),
    ]
    link: Annotated[
        Optional[str],
        Field(description="Optional link to a page where the user can obtain their credentials."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("description", "link", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("vars", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class AuthMethodTerminal(BaseModel):
    id: Annotated[str, Field(description="Unique identifier for this authentication method.")]
    name: Annotated[str, Field(description="Human-readable name of the authentication method.")]
    description: Annotated[
        Optional[str],
        Field(description="Optional description providing more details about this authentication method."),
    ] = None
    args: Annotated[
        Optional[List[str]],
        Field(description="Additional arguments to pass when running the agent binary for terminal auth."),
    ] = None
    env: Annotated[
        Optional[Dict[str, str]],
        Field(description="Additional environment variables to set when running the agent binary for terminal auth."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("description", "env", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("args", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class AuthMethodAgent(BaseModel):
    id: Annotated[str, Field(description="Unique identifier for this authentication method.")]
    name: Annotated[str, Field(description="Human-readable name of the authentication method.")]
    description: Annotated[
        Optional[str],
        Field(description="Optional description providing more details about this authentication method."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("description", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class Implementation(BaseModel):
    name: Annotated[
        str,
        Field(
            description="Intended for programmatic or logical use, but can be used as a display\nname fallback if title isn’t present."
        ),
    ]
    title: Annotated[
        Optional[str],
        Field(
            description="Intended for UI and end-user contexts — optimized to be human-readable\nand easily understood.\n\nIf not provided, the name should be used for display."
        ),
    ] = None
    version: Annotated[
        str,
        Field(
            description='Version of the implementation. Can be displayed to the user or used\nfor debugging or metrics purposes. (e.g. "1.0.0").'
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class AuthenticateResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


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
        Field(alias="apiType", description="Protocol currently used by this provider."),
    ]
    base_url: Annotated[
        str,
        Field(alias="baseUrl", description="Base URL currently used by this provider."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SetProviderResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DisableProviderResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class LogoutResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionMode(BaseModel):
    id: Annotated[
        str,
        Field(description="Stable identifier used to refer to this protocol object in later messages."),
    ]
    name: Annotated[str, Field(description="Human-readable name shown for this protocol object.")]
    description: Annotated[
        Optional[str],
        Field(description="Optional human-readable details shown with this protocol object."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("description", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class SessionConfigSelectOption(BaseModel):
    value: Annotated[str, Field(description="Unique identifier for this option value.")]
    name: Annotated[str, Field(description="Human-readable label for this option value.")]
    description: Annotated[Optional[str], Field(description="Optional description for this option value.")] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("description", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class SessionConfigBoolean(BaseModel):
    current_value: Annotated[
        bool,
        Field(alias="currentValue", description="The current value of the boolean option."),
    ]


class SessionInfo(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId", description="Unique identifier for the session")]
    cwd: Annotated[
        str,
        Field(description="The working directory for this session. Must be an absolute path."),
    ]
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="Additional workspace roots reported for this session. Each path must be absolute.\n\nWhen present, this is the complete ordered additional-root list reported\nby the Agent. Omitted and empty values are equivalent: the response\nreports no additional roots.",
        ),
    ] = None
    title: Annotated[Optional[str], Field(description="Human-readable title for the session")] = None
    updated_at: Annotated[
        Optional[str],
        Field(alias="updatedAt", description="ISO 8601 timestamp of last activity"),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("title", "updated_at", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("additional_directories", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class DeleteSessionResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CloseSessionResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SetSessionModeResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class Usage(BaseModel):
    total_tokens: Annotated[
        int,
        Field(
            alias="totalTokens",
            description="Sum of all token types across session.",
            ge=0,
        ),
    ]
    input_tokens: Annotated[
        int,
        Field(
            alias="inputTokens",
            description="Total input tokens across all turns.",
            ge=0,
        ),
    ]
    output_tokens: Annotated[
        int,
        Field(
            alias="outputTokens",
            description="Total output tokens across all turns.",
            ge=0,
        ),
    ]
    thought_tokens: Annotated[
        Optional[int],
        Field(alias="thoughtTokens", description="Total thought/reasoning tokens", ge=0),
    ] = None
    cached_read_tokens: Annotated[
        Optional[int],
        Field(alias="cachedReadTokens", description="Total cache read tokens.", ge=0),
    ] = None
    cached_write_tokens: Annotated[
        Optional[int],
        Field(alias="cachedWriteTokens", description="Total cache write tokens.", ge=0),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("cached_read_tokens", "cached_write_tokens", "thought_tokens", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class StartNesResponse(BaseModel):
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The session ID for the newly started NES session.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class Position(BaseModel):
    line: Annotated[int, Field(description="Zero-based line number.", ge=0)]
    character: Annotated[
        int,
        Field(description="Zero-based character offset (encoding-dependent).", ge=0),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesJumpSuggestion(BaseModel):
    id: Annotated[str, Field(description="Unique identifier for accept/reject tracking.")]
    uri: Annotated[str, Field(description="The file to navigate to.")]
    position: Annotated[Position, Field(description="The target position within the file.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesRenameSuggestion(BaseModel):
    id: Annotated[str, Field(description="Unique identifier for accept/reject tracking.")]
    uri: Annotated[str, Field(description="The file URI containing the symbol.")]
    position: Annotated[Position, Field(description="The position of the symbol to rename.")]
    new_name: Annotated[str, Field(alias="newName", description="The new name for the symbol.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesSearchAndReplaceSuggestion(BaseModel):
    id: Annotated[str, Field(description="Unique identifier for accept/reject tracking.")]
    uri: Annotated[str, Field(description="The file URI to search within.")]
    search: Annotated[str, Field(description="The text or pattern to find.")]
    replace: Annotated[str, Field(description="The replacement text.")]
    is_regex: Annotated[
        Optional[bool],
        Field(
            alias="isRegex",
            description="Whether `search` is a regular expression. Defaults to `false`.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CloseNesResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class PlanFile(BaseModel):
    plan_id: Annotated[str, Field(alias="planId", description="The plan ID to update.")]
    uri: Annotated[str, Field(description="The URI of the file containing the plan.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class PlanMarkdown(BaseModel):
    plan_id: Annotated[str, Field(alias="planId", description="The plan ID to update.")]
    content: Annotated[str, Field(description="Markdown content for the plan.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class PlanRemoved(BaseModel):
    plan_id: Annotated[str, Field(alias="planId", description="The plan ID to remove.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class UnstructuredCommandInput(BaseModel):
    hint: Annotated[
        str,
        Field(description="A hint to display when the input hasn't been provided yet"),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class _CurrentModeUpdate(BaseModel):
    current_mode_id: Annotated[str, Field(alias="currentModeId", description="The ID of the current mode")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class _SessionInfoUpdate(BaseModel):
    title: Annotated[
        Optional[str],
        Field(description="Human-readable title for the session. Set to null to clear."),
    ] = None
    updated_at: Annotated[
        Optional[str],
        Field(
            alias="updatedAt",
            description="ISO 8601 timestamp of last activity. Set to null to clear.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("title", "updated_at", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class Cost(BaseModel):
    amount: Annotated[float, Field(description="Total cumulative cost for session.")]
    currency: Annotated[str, Field(description='ISO 4217 currency code (e.g., "USD", "EUR").')]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class _UsageUpdate(BaseModel):
    used: Annotated[int, Field(description="Tokens currently in context.", ge=0)]
    size: Annotated[int, Field(description="Total context window size in tokens.", ge=0)]
    cost: Annotated[Optional[Cost], Field(description="Cumulative session cost (optional).")] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("cost", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class CompleteElicitationNotification(BaseModel):
    elicitation_id: Annotated[
        str,
        Field(
            alias="elicitationId",
            description="The ID of the elicitation that completed.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class MessageMcpNotification(BaseModel):
    connection_id: Annotated[
        str,
        Field(
            alias="connectionId",
            description="The MCP-over-ACP connection this message is sent on.",
        ),
    ]
    method: Annotated[str, Field(description="The inner MCP method name.")]
    params: Annotated[
        Optional[Dict[str, Any]],
        Field(
            description="Optional inner MCP params.\n\nIf omitted or set to `null`, the inner MCP message has no params."
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("params", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class FileSystemCapabilities(BaseModel):
    read_text_file: Annotated[
        Optional[bool],
        Field(
            alias="readTextFile",
            description="Whether the Client supports `fs/read_text_file` requests.",
        ),
    ] = False
    write_text_file: Annotated[
        Optional[bool],
        Field(
            alias="writeTextFile",
            description="Whether the Client supports `fs/write_text_file` requests.",
        ),
    ] = False
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("read_text_file", "write_text_file", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: False)


class BooleanConfigOptionCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class PlanCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class AuthCapabilities(BaseModel):
    terminal: Annotated[
        Optional[bool],
        Field(
            description="Whether the client supports `terminal` authentication methods.\n\nWhen `true`, the agent may include `terminal` entries in its authentication methods."
        ),
    ] = False
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("terminal", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: False)


class ElicitationFormCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ElicitationUrlCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesJumpCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesRenameCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesSearchAndReplaceCapabilities(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class AuthenticateRequest(BaseModel):
    method_id: Annotated[
        str,
        Field(
            alias="methodId",
            description="The ID of the authentication method to use.\nMust be one of the methods advertised in the initialize response.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ListProvidersRequest(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SetProviderRequest(BaseModel):
    provider_id: Annotated[str, Field(alias="providerId", description="Provider ID to configure.")]
    api_type: Annotated[
        Union[
            Literal["anthropic"],
            Literal["openai"],
            Literal["azure"],
            Literal["vertex"],
            Literal["bedrock"],
            Dict[str, Any],
        ],
        Field(alias="apiType", description="Protocol type for this provider."),
    ]
    base_url: Annotated[
        str,
        Field(
            alias="baseUrl",
            description="Base URL for requests sent through this provider.",
        ),
    ]
    headers: Annotated[
        Optional[Dict[str, str]],
        Field(
            description="Full headers map for this provider.\nMay include authorization, routing, or other integration-specific headers."
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DisableProviderRequest(BaseModel):
    provider_id: Annotated[str, Field(alias="providerId", description="Provider ID to disable.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class LogoutRequest(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class HttpHeader(BaseModel):
    name: Annotated[str, Field(description="The name of the HTTP header.")]
    value: Annotated[str, Field(description="The value to set for the HTTP header.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class McpServerHttp(BaseModel):
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]
    url: Annotated[str, Field(description="URL to the MCP server.")]
    headers: Annotated[
        List[HttpHeader],
        Field(description="HTTP headers to set when making requests to the MCP server."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class McpServerSse(BaseModel):
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]
    url: Annotated[str, Field(description="URL to the MCP server.")]
    headers: Annotated[
        List[HttpHeader],
        Field(description="HTTP headers to set when making requests to the MCP server."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class McpServerAcp(BaseModel):
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]
    server_id: Annotated[
        str,
        Field(
            alias="serverId",
            description="Unique identifier for this MCP server, generated by the component providing it.\n\nProviders MUST NOT reuse an ID for multiple ACP-transport MCP servers that are visible\non the same ACP connection.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class McpServerStdio(BaseModel):
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]
    command: Annotated[str, Field(description="Absolute path to the MCP server executable.")]
    args: Annotated[
        List[str],
        Field(description="Command-line arguments to pass to the MCP server."),
    ]
    env: Annotated[
        List[EnvVariable],
        Field(description="Environment variables to set when launching the MCP server."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ListSessionsRequest(BaseModel):
    cwd: Annotated[
        Optional[str],
        Field(description="Filter sessions by working directory. Must be an absolute path."),
    ] = None
    cursor: Annotated[
        Optional[str],
        Field(
            description="Opaque cursor token from a previous response's nextCursor field for cursor-based pagination"
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DeleteSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to delete.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CloseSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to close.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SetSessionModeRequest(BaseModel):
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The ID of the session to set the mode for."),
    ]
    mode_id: Annotated[str, Field(alias="modeId", description="The ID of the mode to set.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SetSessionConfigOptionBooleanRequest(BaseModel):
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session to set the configuration option for.",
        ),
    ]
    config_id: Annotated[
        str,
        Field(alias="configId", description="The ID of the configuration option to set."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    value: Annotated[bool, Field(description="The boolean value.")]
    type: Literal["boolean"]


class SetSessionConfigOptionSelectRequest(BaseModel):
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session to set the configuration option for.",
        ),
    ]
    config_id: Annotated[
        str,
        Field(alias="configId", description="The ID of the configuration option to set."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    value: Annotated[str, Field(description="The value ID.")]


class WorkspaceFolder(BaseModel):
    uri: Annotated[str, Field(description="The URI of the folder.")]
    name: Annotated[str, Field(description="The display name of the folder.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesRepository(BaseModel):
    name: Annotated[str, Field(description="The repository name.")]
    owner: Annotated[str, Field(description="The repository owner.")]
    remote_url: Annotated[str, Field(alias="remoteUrl", description="The remote URL of the repository.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesRecentFile(BaseModel):
    uri: Annotated[str, Field(description="The URI of the file.")]
    language_id: Annotated[str, Field(alias="languageId", description="The language identifier.")]
    text: Annotated[str, Field(description="The full text content of the file.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesExcerpt(BaseModel):
    start_line: Annotated[
        int,
        Field(
            alias="startLine",
            description="The start line of the excerpt (zero-based).",
            ge=0,
        ),
    ]
    end_line: Annotated[
        int,
        Field(
            alias="endLine",
            description="The end line of the excerpt (zero-based).",
            ge=0,
        ),
    ]
    text: Annotated[str, Field(description="The text content of the excerpt.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesEditHistoryEntry(BaseModel):
    uri: Annotated[str, Field(description="The URI of the edited file.")]
    diff: Annotated[str, Field(description="A diff representing the edit.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesUserAction(BaseModel):
    action: Annotated[
        str,
        Field(description='The kind of action (e.g., "insertChar", "cursorMovement").'),
    ]
    uri: Annotated[str, Field(description="The URI of the file where the action occurred.")]
    position: Annotated[Position, Field(description="The position where the action occurred.")]
    timestamp_ms: Annotated[
        int,
        Field(
            alias="timestampMs",
            description="Timestamp in milliseconds since epoch.",
            ge=0,
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CloseNesRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the NES session to close.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class WriteTextFileResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ReadTextFileResponse(BaseModel):
    content: Annotated[str, Field(description="Content payload returned by this response.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DeniedOutcome(BaseModel):
    outcome: Literal["cancelled"]


class SelectedPermissionOutcome(BaseModel):
    option_id: Annotated[
        str,
        Field(alias="optionId", description="The ID of the option the user selected."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CreateTerminalResponse(BaseModel):
    terminal_id: Annotated[
        str,
        Field(
            alias="terminalId",
            description="The unique identifier for the created terminal.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class TerminalExitStatus(BaseModel):
    exit_code: Annotated[
        Optional[int],
        Field(
            alias="exitCode",
            description="The process exit code (may be null if terminated by signal).",
            ge=0,
        ),
    ] = None
    signal: Annotated[
        Optional[str],
        Field(description="The signal that terminated the process (may be null if exited normally)."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("exit_code", "signal", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ReleaseTerminalResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class WaitForTerminalExitResponse(BaseModel):
    exit_code: Annotated[
        Optional[int],
        Field(
            alias="exitCode",
            description="The process exit code (may be null if terminated by signal).",
            ge=0,
        ),
    ] = None
    signal: Annotated[
        Optional[str],
        Field(description="The signal that terminated the process (may be null if exited normally)."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("exit_code", "signal", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class KillTerminalResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DeclineElicitationResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    action: Literal["decline"]


class CancelElicitationResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    action: Literal["cancel"]


class OtherElicitationResponse(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    action: Annotated[
        str,
        Field(
            description="Custom or future elicitation action.\n\nValues beginning with `_` are reserved for implementation-specific\nextensions. Unknown values that do not begin with `_` are reserved for\nfuture ACP variants."
        ),
    ]

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
    root: Annotated[
        Union[str, int, float, bool, List[str]],
        Field(description="Allowed wire representations for [`ElicitationContentValue`]."),
    ]


class ElicitationAcceptAction(BaseModel):
    content: Annotated[
        Optional[Dict[str, Any]],
        Field(description="The user-provided content, if any, as an object matching the requested schema."),
    ] = None


class ConnectMcpResponse(BaseModel):
    connection_id: Annotated[
        str,
        Field(
            alias="connectionId",
            description="The unique identifier for this MCP-over-ACP connection.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DisconnectMcpResponse(BaseModel):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CancelNotification(BaseModel):
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session to cancel operations for.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DidOpenDocumentNotification(BaseModel):
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    uri: Annotated[str, Field(description="The URI of the opened document.")]
    language_id: Annotated[
        str,
        Field(
            alias="languageId",
            description='The language identifier of the document (e.g., "rust", "python").',
        ),
    ]
    version: Annotated[int, Field(description="The version number of the document.")]
    text: Annotated[str, Field(description="The full text content of the document.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DidCloseDocumentNotification(BaseModel):
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    uri: Annotated[str, Field(description="The URI of the closed document.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DidSaveDocumentNotification(BaseModel):
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    uri: Annotated[str, Field(description="The URI of the saved document.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class AcceptNesNotification(BaseModel):
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    id: Annotated[str, Field(description="The ID of the accepted suggestion.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CancelRequestNotification(BaseModel):
    request_id: Annotated[
        Optional[Union[int, str]],
        Field(alias="requestId", description="The ID of the request to cancel."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ]


class WriteTextFileRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    path: Annotated[str, Field(description="Absolute path to the file to write.")]
    content: Annotated[str, Field(description="The text content to write to the file.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class FileEditToolCallContent(Diff):
    type: Literal["diff"]


class TerminalToolCallContent(Terminal):
    type: Literal["terminal"]


class Annotations(BaseModel):
    audience: Annotated[
        Optional[List[str]],
        Field(description="Intended recipients for this content, such as the user or assistant."),
    ] = None
    last_modified: Annotated[
        Optional[str],
        Field(
            alias="lastModified",
            description="Timestamp indicating when the underlying resource was last modified.",
        ),
    ] = None
    priority: Annotated[
        Optional[float],
        Field(description="Relative importance of this content when clients choose what to surface."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("last_modified", "priority", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("audience", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class TextContent(BaseModel):
    annotations: Annotated[
        Optional[Annotations],
        Field(description="Optional annotations that help clients decide how to display or route this content."),
    ] = None
    text: Annotated[str, Field(description="Text payload carried by this content block.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("annotations", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ImageContent(BaseModel):
    annotations: Annotated[
        Optional[Annotations],
        Field(description="Optional annotations that help clients decide how to display or route this content."),
    ] = None
    data: Annotated[str, Field(description="Base64-encoded media payload.")]
    mime_type: Annotated[
        str,
        Field(
            alias="mimeType",
            description="MIME type describing the encoded media payload.",
        ),
    ]
    uri: Annotated[
        Optional[str],
        Field(description="URI associated with this resource or media payload."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("annotations", "uri", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class AudioContent(BaseModel):
    annotations: Annotated[
        Optional[Annotations],
        Field(description="Optional annotations that help clients decide how to display or route this content."),
    ] = None
    data: Annotated[str, Field(description="Base64-encoded media payload.")]
    mime_type: Annotated[
        str,
        Field(
            alias="mimeType",
            description="MIME type describing the encoded media payload.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("annotations", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ResourceLink(BaseModel):
    annotations: Annotated[
        Optional[Annotations],
        Field(description="Optional annotations that help clients decide how to display or route this content."),
    ] = None
    description: Annotated[
        Optional[str],
        Field(description="Optional human-readable details shown with this protocol object."),
    ] = None
    mime_type: Annotated[
        Optional[str],
        Field(
            alias="mimeType",
            description="MIME type describing the encoded media payload.",
        ),
    ] = None
    name: Annotated[str, Field(description="Human-readable name shown for this protocol object.")]
    size: Annotated[
        Optional[int],
        Field(description="Optional size of the linked resource in bytes, if known."),
    ] = None
    title: Annotated[Optional[str], Field(description="Optional display title for end-user UI.")] = None
    uri: Annotated[str, Field(description="URI associated with this resource or media payload.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("annotations", "description", "mime_type", "size", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class EmbeddedResource(BaseModel):
    annotations: Annotated[
        Optional[Annotations],
        Field(description="Optional annotations that help clients decide how to display or route this content."),
    ] = None
    resource: Annotated[
        Union[TextResourceContents, BlobResourceContents],
        Field(description="Embedded resource payload, either text or binary data."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("annotations", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class PermissionOption(BaseModel):
    option_id: Annotated[
        str,
        Field(
            alias="optionId",
            description="Unique identifier for this permission option.",
        ),
    ]
    name: Annotated[str, Field(description="Human-readable label to display to the user.")]
    kind: Annotated[PermissionOptionKind, Field(description="Hint about the nature of this permission option.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CreateTerminalRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    command: Annotated[str, Field(description="The command to execute.")]
    args: Annotated[Optional[List[str]], Field(description="Array of command arguments.")] = None
    env: Annotated[
        Optional[List[EnvVariable]],
        Field(description="Environment variables for the command."),
    ] = None
    cwd: Annotated[
        Optional[str],
        Field(description="Working directory for the command. Must be an absolute path."),
    ] = None
    output_byte_limit: Annotated[
        Optional[int],
        Field(
            alias="outputByteLimit",
            description="Maximum number of output bytes to retain.\n\nWhen the limit is exceeded, the Client truncates from the beginning of the output\nto stay within the limit.\n\nThe Client MUST ensure truncation happens at a character boundary to maintain valid\nstring output, even if this means the retained output is slightly less than the\nspecified limit.",
            ge=0,
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
    message: Annotated[
        str,
        Field(description="A human-readable message describing what input is needed."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    mode: Literal["url"]
    elicitation_id: Annotated[
        str,
        Field(
            alias="elicitationId",
            description="The unique identifier for this elicitation.",
        ),
    ]
    url: Annotated[AnyUrl, Field(description="The URL to direct the user to.")]


class CreateUrlRequestElicitationRequest(ElicitationRequestScope):
    message: Annotated[
        str,
        Field(description="A human-readable message describing what input is needed."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    mode: Literal["url"]
    elicitation_id: Annotated[
        str,
        Field(
            alias="elicitationId",
            description="The unique identifier for this elicitation.",
        ),
    ]
    url: Annotated[AnyUrl, Field(description="The URL to direct the user to.")]


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
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    min_items: Annotated[
        Optional[int],
        Field(alias="minItems", description="Minimum number of items to select.", ge=0),
    ] = None
    max_items: Annotated[
        Optional[int],
        Field(alias="maxItems", description="Maximum number of items to select.", ge=0),
    ] = None
    items: Annotated[
        Union[StringMultiSelectItems, OtherMultiSelectItems, TitledMultiSelectItems],
        Field(description="The items definition describing allowed values."),
    ]
    default: Annotated[Optional[List[str]], Field(description="Default selected values.")] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("description", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("default", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ConnectMcpRequest(BaseModel):
    server_id: Annotated[
        str,
        Field(
            alias="serverId",
            description="The ACP MCP server ID that was provided by the component declaring the MCP server.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class MessageMcpRequest(BaseModel):
    connection_id: Annotated[
        str,
        Field(
            alias="connectionId",
            description="The MCP-over-ACP connection this message is sent on.",
        ),
    ]
    method: Annotated[str, Field(description="The inner MCP method name.")]
    params: Annotated[
        Optional[Dict[str, Any]],
        Field(
            description="Optional inner MCP params.\n\nIf omitted or set to `null`, the inner MCP message has no params."
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionCapabilities(BaseModel):
    list: Annotated[
        Optional[SessionListCapabilities],
        Field(
            description="Whether the agent supports `session/list`.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports listing sessions."
        ),
    ] = None
    delete: Annotated[
        Optional[SessionDeleteCapabilities],
        Field(
            description="Whether the agent supports `session/delete`.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports deleting sessions from `session/list`."
        ),
    ] = None
    additional_directories: Annotated[
        Optional[SessionAdditionalDirectoriesCapabilities],
        Field(
            alias="additionalDirectories",
            description="Whether the agent supports `additionalDirectories` on supported session lifecycle requests.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports `additionalDirectories` on\nsupported session lifecycle requests.\n\nAgents that also support `session/list` may return\n`SessionInfo.additionalDirectories` to report the complete ordered\nadditional-root list associated with a listed session.",
        ),
    ] = None
    fork: Annotated[
        Optional[SessionForkCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nWhether the agent supports `session/fork`.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports forking sessions."
        ),
    ] = None
    resume: Annotated[
        Optional[SessionResumeCapabilities],
        Field(
            description="Whether the agent supports `session/resume`.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports resuming sessions."
        ),
    ] = None
    close: Annotated[
        Optional[SessionCloseCapabilities],
        Field(
            description="Whether the agent supports `session/close`.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports closing sessions."
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("additional_directories", "close", "delete", "fork", "list", "resume", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class AgentAuthCapabilities(BaseModel):
    logout: Annotated[
        Optional[LogoutCapabilities],
        Field(
            description="Whether the agent supports the logout method.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports the logout method."
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("logout", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NesDocumentDidChangeCapabilities(BaseModel):
    sync_kind: Annotated[
        str,
        Field(
            alias="syncKind",
            description='The sync kind the agent wants: `"full"` or `"incremental"`.',
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesContextCapabilities(BaseModel):
    recent_files: Annotated[
        Optional[NesRecentFilesCapabilities],
        Field(
            alias="recentFiles",
            description="Whether the agent wants recent files context.",
        ),
    ] = None
    related_snippets: Annotated[
        Optional[NesRelatedSnippetsCapabilities],
        Field(
            alias="relatedSnippets",
            description="Whether the agent wants related snippets context.",
        ),
    ] = None
    edit_history: Annotated[
        Optional[NesEditHistoryCapabilities],
        Field(
            alias="editHistory",
            description="Whether the agent wants edit history context.",
        ),
    ] = None
    user_actions: Annotated[
        Optional[NesUserActionsCapabilities],
        Field(
            alias="userActions",
            description="Whether the agent wants user actions context.",
        ),
    ] = None
    open_files: Annotated[
        Optional[NesOpenFilesCapabilities],
        Field(alias="openFiles", description="Whether the agent wants open files context."),
    ] = None
    diagnostics: Annotated[
        Optional[NesDiagnosticsCapabilities],
        Field(description="Whether the agent wants diagnostics context."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
    provider_id: Annotated[
        str,
        Field(
            alias="providerId",
            description='Provider identifier, for example "main" or "openai".',
        ),
    ]
    supported: Annotated[
        List[
            Union[
                Literal["anthropic"],
                Literal["openai"],
                Literal["azure"],
                Literal["vertex"],
                Literal["bedrock"],
                Dict[str, Any],
            ]
        ],
        Field(description="Supported protocol types for this provider."),
    ]
    required: Annotated[
        bool,
        Field(
            description="Whether this provider is mandatory and cannot be disabled via `providers/disable`.\nIf true, clients must not call `providers/disable` for this provider ID."
        ),
    ]
    current: Annotated[
        Optional[ProviderCurrentConfig],
        Field(description="Current effective non-secret routing config.\nNull or omitted means provider is disabled."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("supported", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class SessionModeState(BaseModel):
    current_mode_id: Annotated[
        str,
        Field(alias="currentModeId", description="The current mode the Agent is in."),
    ]
    available_modes: Annotated[
        List[SessionMode],
        Field(
            alias="availableModes",
            description="The set of modes that the Agent can operate in",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("available_modes", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class SessionConfigOptionBoolean(SessionConfigBoolean):
    id: Annotated[str, Field(description="Unique identifier for the configuration option.")]
    name: Annotated[str, Field(description="Human-readable label for the option.")]
    description: Annotated[
        Optional[str],
        Field(description="Optional description for the Client to display to the user."),
    ] = None
    category: Annotated[
        Optional[
            Union[
                Literal["mode"],
                Literal["model"],
                Literal["model_config"],
                Literal["thought_level"],
                Dict[str, Any],
            ]
        ],
        Field(description="Optional semantic category for this option (UX only)."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    type: Literal["boolean"]

    @field_validator("category", "description", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class SessionConfigSelectGroup(BaseModel):
    group: Annotated[str, Field(description="Unique identifier for this group.")]
    name: Annotated[str, Field(description="Human-readable label for this group.")]
    options: Annotated[
        List[SessionConfigSelectOption],
        Field(description="The set of option values in this group."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("options", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ListSessionsResponse(BaseModel):
    sessions: Annotated[List[SessionInfo], Field(description="Array of session information objects")]
    next_cursor: Annotated[
        Optional[str],
        Field(
            alias="nextCursor",
            description="Opaque cursor token. If present, pass this in the next request's cursor parameter\nto fetch the next page. If absent, there are no more results.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("next_cursor", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("sessions", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class PromptResponse(BaseModel):
    stop_reason: Annotated[
        StopReason,
        Field(
            alias="stopReason",
            description="Indicates why the agent stopped processing the turn.",
        ),
    ]
    usage: Annotated[
        Optional[Usage],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nToken usage for this turn (optional)."
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
    start: Annotated[Position, Field(description="The start position (inclusive).")]
    end: Annotated[Position, Field(description="The end position (exclusive).")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class Error(BaseModel):
    code: Annotated[
        Union[
            Literal[-32700],
            Literal[-32600],
            Literal[-32601],
            Literal[-32602],
            Literal[-32603],
            Literal[-32800],
            Literal[-32000],
            Literal[-32002],
            int,
        ],
        Field(
            description="A number indicating the error type that occurred.\nThis must be an integer as defined in the JSON-RPC specification."
        ),
    ]
    message: Annotated[
        str,
        Field(
            description="A string providing a short description of the error.\nThe message should be limited to a concise single sentence."
        ),
    ]
    data: Annotated[
        Optional[Any],
        Field(
            description="Optional primitive or structured value that contains additional information about the error.\nThis may include debugging information or context-specific details."
        ),
    ] = None

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
    content: Annotated[
        str,
        Field(description="Human-readable description of what this task aims to accomplish."),
    ]
    priority: Annotated[
        PlanEntryPriority,
        Field(
            description="The relative importance of this task.\nUsed to indicate which tasks are most critical to the overall goal."
        ),
    ]
    status: Annotated[PlanEntryStatus, Field(description="Current execution status of this task.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class Plan(BaseModel):
    entries: Annotated[
        List[PlanEntry],
        Field(
            description="The list of tasks to be accomplished.\n\nWhen updating a plan, the agent must send a complete list of all entries\nwith their current status. The client replaces the entire plan with each update."
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("entries", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class PlanUpdateFile(PlanFile):
    type: Literal["file"]


class PlanUpdateMarkdown(PlanMarkdown):
    type: Literal["markdown"]


class PlanItems(BaseModel):
    plan_id: Annotated[str, Field(alias="planId", description="The plan ID to update.")]
    entries: Annotated[
        List[PlanEntry],
        Field(
            description="The list of tasks to be accomplished.\n\nWhen updating an item-based plan, the agent must send a complete list of all entries\nwith their current status. The client replaces that plan with each update."
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("entries", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class AvailableCommandInput(RootModel[UnstructuredCommandInput]):
    root: Annotated[
        UnstructuredCommandInput,
        Field(description="The input specification for a command."),
    ]


class SessionConfigOptionsCapabilities(BaseModel):
    boolean: Annotated[
        Optional[BooleanConfigOptionCapabilities],
        Field(
            description='Whether the client supports boolean session configuration options.\n\nOptional. Omitted or `null` both mean the client does not advertise support.\nSupplying `{}` means agents may include `type: "boolean"` entries in\n`configOptions`, and the client may send `session/set_config_option`\nrequests with `type: "boolean"` and a boolean `value`.'
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("boolean", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ElicitationCapabilities(BaseModel):
    form: Annotated[
        Optional[ElicitationFormCapabilities],
        Field(
            description="Whether the client supports form-based elicitation.\n\nOptional. Omitted or `null` both mean the client does not advertise support.\nSupplying `{}` means the client supports form-based elicitation."
        ),
    ] = None
    url: Annotated[
        Optional[ElicitationUrlCapabilities],
        Field(
            description="Whether the client supports URL-based elicitation.\n\nOptional. Omitted or `null` both mean the client does not advertise support.\nSupplying `{}` means the client supports URL-based elicitation."
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("form", "url", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ClientNesCapabilities(BaseModel):
    jump: Annotated[
        Optional[NesJumpCapabilities],
        Field(description="Whether the client supports the `jump` suggestion kind."),
    ] = None
    rename: Annotated[
        Optional[NesRenameCapabilities],
        Field(description="Whether the client supports the `rename` suggestion kind."),
    ] = None
    search_and_replace: Annotated[
        Optional[NesSearchAndReplaceCapabilities],
        Field(
            alias="searchAndReplace",
            description="Whether the client supports the `searchAndReplace` suggestion kind.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
        Field(
            alias="mcpServers",
            description="List of MCP servers to connect to for this session.",
        ),
    ]
    cwd: Annotated[
        str,
        Field(description="The working directory for this session. Must be an absolute path."),
    ]
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="Additional workspace roots to activate for this session. Each path must be absolute.\n\nWhen omitted or empty, no additional roots are activated. When non-empty,\nthis is the complete resulting additional-root list for the loaded\nsession. It may differ from any previously used or reported list as long as\nthe request `cwd` matches the session's `cwd`.",
        ),
    ] = None
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to load.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("additional_directories", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)

    @field_validator("mcp_servers", mode="wrap")
    @classmethod
    def _skip_invalid_items_1(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ForkSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to fork.")]
    cwd: Annotated[
        str,
        Field(description="The working directory for this session. Must be an absolute path."),
    ]
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="Additional workspace roots to activate for this session. Each path must be absolute.\n\nWhen omitted or empty, no additional roots are activated. When non-empty,\nthis is the complete resulting additional-root list for the forked\nsession.",
        ),
    ] = None
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]]],
        Field(
            alias="mcpServers",
            description="List of MCP servers to connect to for this session.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("additional_directories", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)

    @field_validator("mcp_servers", mode="wrap")
    @classmethod
    def _skip_invalid_items_1(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ResumeSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to resume.")]
    cwd: Annotated[
        str,
        Field(description="The working directory for this session. Must be an absolute path."),
    ]
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="Additional workspace roots to activate for this session. Each path must be absolute.\n\nWhen omitted or empty, no additional roots are activated. When non-empty,\nthis is the complete resulting additional-root list for the resumed\nsession. It may differ from any previously used or reported list as long as\nthe request `cwd` matches the session's `cwd`.",
        ),
    ] = None
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]]],
        Field(
            alias="mcpServers",
            description="List of MCP servers to connect to for this session.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("additional_directories", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)

    @field_validator("mcp_servers", mode="wrap")
    @classmethod
    def _skip_invalid_items_1(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class StartNesRequest(BaseModel):
    workspace_uri: Annotated[
        Optional[str],
        Field(alias="workspaceUri", description="The root URI of the workspace."),
    ] = None
    workspace_folders: Annotated[
        Optional[List[WorkspaceFolder]],
        Field(alias="workspaceFolders", description="The workspace folders."),
    ] = None
    repository: Annotated[
        Optional[NesRepository],
        Field(description="Repository metadata, if the workspace is a git repository."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("repository", "workspace_uri", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NesRelatedSnippet(BaseModel):
    uri: Annotated[str, Field(description="The URI of the file containing the snippets.")]
    excerpts: Annotated[List[NesExcerpt], Field(description="The code excerpts.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesOpenFile(BaseModel):
    uri: Annotated[str, Field(description="The URI of the file.")]
    language_id: Annotated[str, Field(alias="languageId", description="The language identifier.")]
    visible_range: Annotated[
        Optional[Range],
        Field(alias="visibleRange", description="The visible range in the editor, if any."),
    ] = None
    last_focused_ms: Annotated[
        Optional[int],
        Field(
            alias="lastFocusedMs",
            description="Timestamp in milliseconds since epoch of when the file was last focused.",
            ge=0,
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("last_focused_ms", "visible_range", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NesDiagnostic(BaseModel):
    uri: Annotated[str, Field(description="The URI of the file containing the diagnostic.")]
    range: Annotated[Range, Field(description="The range of the diagnostic.")]
    severity: Annotated[str, Field(description="The severity of the diagnostic.")]
    message: Annotated[str, Field(description="The diagnostic message.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ClientErrorMessage(BaseModel):
    id: Annotated[
        Optional[Union[int, str]],
        Field(description="The id of the request this response answers."),
    ]
    error: Annotated[Error, Field(description="Method-specific error data.")]


class AllowedOutcome(SelectedPermissionOutcome):
    outcome: Literal["selected"]


class TerminalOutputResponse(BaseModel):
    output: Annotated[str, Field(description="The terminal output captured so far.")]
    truncated: Annotated[bool, Field(description="Whether the output was truncated due to byte limits.")]
    exit_status: Annotated[
        Optional[TerminalExitStatus],
        Field(alias="exitStatus", description="Exit status if the command has completed."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("exit_status", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class AcceptElicitationResponse(ElicitationAcceptAction):
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    action: Literal["accept"]


class TextDocumentContentChangeEvent(BaseModel):
    range: Annotated[
        Optional[Range],
        Field(description="The range of the document that changed. If `None`, the entire content is replaced."),
    ] = None
    text: Annotated[
        str,
        Field(description="The new text for the range, or the full document content if `range` is `None`."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DidFocusDocumentNotification(BaseModel):
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    uri: Annotated[str, Field(description="The URI of the focused document.")]
    version: Annotated[int, Field(description="The version number of the document.")]
    position: Annotated[Position, Field(description="The current cursor position.")]
    visible_range: Annotated[
        Range,
        Field(
            alias="visibleRange",
            description="The portion of the file currently visible in the editor viewport.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class RejectNesNotification(BaseModel):
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    id: Annotated[str, Field(description="The ID of the rejected suggestion.")]
    reason: Annotated[Optional[str], Field(description="The reason for rejection.")] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
        Field(description="The actual content block.", discriminator="type"),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ElicitationMultiSelectPropertySchema(MultiSelectPropertySchema):
    type: Literal["array"]


class AgentErrorMessage(BaseModel):
    id: Annotated[
        Optional[Union[int, str]],
        Field(description="The id of the request this response answers."),
    ]
    error: Annotated[Error, Field(description="Method-specific error data.")]


class NesDocumentEventCapabilities(BaseModel):
    did_open: Annotated[
        Optional[NesDocumentDidOpenCapabilities],
        Field(
            alias="didOpen",
            description="Whether the agent wants `document/didOpen` events.",
        ),
    ] = None
    did_change: Annotated[
        Optional[NesDocumentDidChangeCapabilities],
        Field(
            alias="didChange",
            description="Whether the agent wants `document/didChange` events, and the sync kind.",
        ),
    ] = None
    did_close: Annotated[
        Optional[NesDocumentDidCloseCapabilities],
        Field(
            alias="didClose",
            description="Whether the agent wants `document/didClose` events.",
        ),
    ] = None
    did_save: Annotated[
        Optional[NesDocumentDidSaveCapabilities],
        Field(
            alias="didSave",
            description="Whether the agent wants `document/didSave` events.",
        ),
    ] = None
    did_focus: Annotated[
        Optional[NesDocumentDidFocusCapabilities],
        Field(
            alias="didFocus",
            description="Whether the agent wants `document/didFocus` events.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("did_change", "did_close", "did_focus", "did_open", "did_save", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ListProvidersResponse(BaseModel):
    providers: Annotated[
        List[ProviderInfo],
        Field(description="Configurable providers with current routing info suitable for UI display."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionConfigSelect(BaseModel):
    current_value: Annotated[str, Field(alias="currentValue", description="The currently selected value.")]
    options: Annotated[
        Union[List[SessionConfigSelectOption], List[SessionConfigSelectGroup]],
        Field(description="The set of selectable options."),
    ]


class NesTextEdit(BaseModel):
    range: Annotated[Range, Field(description="The range to replace.")]
    new_text: Annotated[str, Field(alias="newText", description="The replacement text.")]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesEditSuggestion(BaseModel):
    id: Annotated[str, Field(description="Unique identifier for accept/reject tracking.")]
    uri: Annotated[str, Field(description="The URI of the file to edit.")]
    edits: Annotated[List[NesTextEdit], Field(description="The text edits to apply.")]
    cursor_position: Annotated[
        Optional[Position],
        Field(
            alias="cursorPosition",
            description="Optional suggested cursor position after applying edits.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
        Field(description="A single item of content", discriminator="type"),
    ]
    message_id: Annotated[
        Optional[str],
        Field(
            alias="messageId",
            description="A unique identifier for the message this chunk belongs to.\n\nAll chunks belonging to the same message share the same `messageId`.\nA change in `messageId` indicates a new message has started.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("message_id", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class PlanUpdateItems(PlanItems):
    type: Literal["items"]


class PlanUpdate(BaseModel):
    plan: Annotated[
        Union[PlanUpdateItems, PlanUpdateFile, PlanUpdateMarkdown],
        Field(description="The updated plan content.", discriminator="type"),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class AvailableCommand(BaseModel):
    name: Annotated[
        str,
        Field(description="Command name (e.g., `create_plan`, `research_codebase`)."),
    ]
    description: Annotated[str, Field(description="Human-readable description of what the command does.")]
    input: Annotated[
        Optional[AvailableCommandInput],
        Field(description="Input for the command if required"),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("input", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class _AvailableCommandsUpdate(BaseModel):
    available_commands: Annotated[
        List[AvailableCommand],
        Field(alias="availableCommands", description="Commands the agent can execute"),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("available_commands", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ClientSessionCapabilities(BaseModel):
    config_options: Annotated[
        Optional[SessionConfigOptionsCapabilities],
        Field(
            alias="configOptions",
            description="Config option capabilities supported by the client.\n\nOmitted or `null` both mean the client does not advertise support for any\nconfig option extensions.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("config_options", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NewSessionRequest(BaseModel):
    cwd: Annotated[
        str,
        Field(description="The working directory for this session. Must be an absolute path."),
    ]
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="Additional workspace roots for this session. Each path must be absolute.\n\nThese expand the session's filesystem scope without changing `cwd`, which\nremains the base for relative paths. When omitted or empty, no\nadditional roots are activated for the new session.",
        ),
    ] = None
    mcp_servers: Annotated[
        List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]],
        Field(
            alias="mcpServers",
            description="List of MCP (Model Context Protocol) servers the agent should connect to.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("additional_directories", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)

    @field_validator("mcp_servers", mode="wrap")
    @classmethod
    def _skip_invalid_items_1(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class PromptRequest(BaseModel):
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session to send this user message to",
        ),
    ]
    prompt: Annotated[
        List[
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
        ],
        Field(
            description="The blocks of content that compose the user's message.\n\nAs a baseline, the Agent MUST support [`ContentBlock::Text`] and [`ContentBlock::ResourceLink`],\nwhile other variants are optionally enabled via [`PromptCapabilities`].\n\nThe Client MUST adapt its interface according to [`PromptCapabilities`].\n\nThe client MAY include referenced pieces of context as either\n[`ContentBlock::Resource`] or [`ContentBlock::ResourceLink`].\n\nWhen available, [`ContentBlock::Resource`] is preferred\nas it avoids extra round-trips and allows the message to include\npieces of context from sources the agent may not have access to."
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesSuggestContext(BaseModel):
    recent_files: Annotated[
        Optional[List[NesRecentFile]],
        Field(alias="recentFiles", description="Recently accessed files."),
    ] = None
    related_snippets: Annotated[
        Optional[List[NesRelatedSnippet]],
        Field(alias="relatedSnippets", description="Related code snippets."),
    ] = None
    edit_history: Annotated[
        Optional[List[NesEditHistoryEntry]],
        Field(alias="editHistory", description="Recent edit history."),
    ] = None
    user_actions: Annotated[
        Optional[List[NesUserAction]],
        Field(
            alias="userActions",
            description="Recent user actions (typing, navigation, etc.).",
        ),
    ] = None
    open_files: Annotated[
        Optional[List[NesOpenFile]],
        Field(alias="openFiles", description="Currently open files in the editor."),
    ] = None
    diagnostics: Annotated[
        Optional[List[NesDiagnostic]],
        Field(description="Current diagnostics (errors, warnings)."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class RequestPermissionResponse(BaseModel):
    outcome: Annotated[
        Union[DeniedOutcome, AllowedOutcome],
        Field(
            description="The user's decision on the permission request.",
            discriminator="outcome",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DidChangeDocumentNotification(BaseModel):
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    uri: Annotated[str, Field(description="The URI of the changed document.")]
    version: Annotated[int, Field(description="The new version number of the document.")]
    content_changes: Annotated[
        List[TextDocumentContentChangeEvent],
        Field(alias="contentChanges", description="The content changes."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("content_changes", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ContentToolCallContent(Content):
    type: Literal["content"]


class ElicitationSchema(BaseModel):
    type: Annotated[Optional[str], Field(description='Type discriminator. Always `"object"`.')] = "object"
    title: Annotated[Optional[str], Field(description="Optional title for the schema.")] = None
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
        Field(
            description="Property definitions (must be primitive types).",
            validate_default=True,
        ),
    ] = {}
    required: Annotated[Optional[List[str]], Field(description="List of required property names.")] = None
    description: Annotated[
        Optional[str],
        Field(description="Optional description of what this schema represents."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("type", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: "object")

    @field_validator("description", "title", mode="wrap")
    @classmethod
    def _salvage_on_error_1(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ElicitationFormSessionMode(ElicitationSessionScope):
    requested_schema: Annotated[
        ElicitationSchema,
        Field(
            alias="requestedSchema",
            description="A JSON Schema describing the form fields to present to the user.",
        ),
    ]


class ElicitationFormRequestMode(ElicitationRequestScope):
    requested_schema: Annotated[
        ElicitationSchema,
        Field(
            alias="requestedSchema",
            description="A JSON Schema describing the form fields to present to the user.",
        ),
    ]


class ElicitationFormMode(RootModel[Union[ElicitationFormSessionMode, ElicitationFormRequestMode]]):
    root: Annotated[
        Union[ElicitationFormSessionMode, ElicitationFormRequestMode],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nForm-based elicitation mode where the client renders a form from the provided schema."
        ),
    ]


class NesEventCapabilities(BaseModel):
    document: Annotated[
        Optional[NesDocumentEventCapabilities],
        Field(description="Document event capabilities."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("document", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class SessionConfigOptionSelect(SessionConfigSelect):
    id: Annotated[str, Field(description="Unique identifier for the configuration option.")]
    name: Annotated[str, Field(description="Human-readable label for the option.")]
    description: Annotated[
        Optional[str],
        Field(description="Optional description for the Client to display to the user."),
    ] = None
    category: Annotated[
        Optional[
            Union[
                Literal["mode"],
                Literal["model"],
                Literal["model_config"],
                Literal["thought_level"],
                Dict[str, Any],
            ]
        ],
        Field(description="Optional semantic category for this option (UX only)."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    type: Literal["select"]

    @field_validator("category", "description", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class LoadSessionResponse(BaseModel):
    modes: Annotated[
        Optional[SessionModeState],
        Field(
            description="Initial mode state if supported by the Agent\n\nSee protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)"
        ),
    ] = None
    config_options: Annotated[
        Optional[
            List[
                Annotated[
                    Union[SessionConfigOptionSelect, SessionConfigOptionBoolean],
                    Field(discriminator="type"),
                ]
            ]
        ],
        Field(
            alias="configOptions",
            description="Initial session configuration options if supported by the Agent.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("modes", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("config_options", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ForkSessionResponse(BaseModel):
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="Unique identifier for the newly created forked session.",
        ),
    ]
    modes: Annotated[
        Optional[SessionModeState],
        Field(
            description="Initial mode state if supported by the Agent\n\nSee protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)"
        ),
    ] = None
    config_options: Annotated[
        Optional[
            List[
                Annotated[
                    Union[SessionConfigOptionSelect, SessionConfigOptionBoolean],
                    Field(discriminator="type"),
                ]
            ]
        ],
        Field(
            alias="configOptions",
            description="Initial session configuration options if supported by the Agent.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("modes", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("config_options", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ResumeSessionResponse(BaseModel):
    modes: Annotated[
        Optional[SessionModeState],
        Field(
            description="Initial mode state if supported by the Agent\n\nSee protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)"
        ),
    ] = None
    config_options: Annotated[
        Optional[
            List[
                Annotated[
                    Union[SessionConfigOptionSelect, SessionConfigOptionBoolean],
                    Field(discriminator="type"),
                ]
            ]
        ],
        Field(
            alias="configOptions",
            description="Initial session configuration options if supported by the Agent.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
        Field(
            alias="configOptions",
            description="The full set of configuration options and their current values.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
    tool_call_id: Annotated[
        str,
        Field(
            alias="toolCallId",
            description="Unique identifier for this tool call within the session.",
        ),
    ]
    title: Annotated[
        str,
        Field(description="Human-readable title describing what the tool is doing."),
    ]
    kind: Annotated[
        Optional[ToolKind],
        Field(
            description="The category of tool being invoked.\nHelps clients choose appropriate icons and UI treatment."
        ),
    ] = None
    status: Annotated[Optional[ToolCallStatus], Field(description="Current execution status of the tool call.")] = None
    content: Annotated[
        Optional[
            List[
                Annotated[
                    Union[ContentToolCallContent, FileEditToolCallContent, TerminalToolCallContent],
                    Field(discriminator="type"),
                ]
            ]
        ],
        Field(description="Content produced by the tool call."),
    ] = None
    locations: Annotated[
        Optional[List[ToolCallLocation]],
        Field(description='File locations affected by this tool call.\nEnables "follow-along" features in clients.'),
    ] = None
    raw_input: Annotated[
        Optional[Any],
        Field(alias="rawInput", description="Raw input parameters sent to the tool."),
    ] = None
    raw_output: Annotated[
        Optional[Any],
        Field(alias="rawOutput", description="Raw output returned by the tool."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
        Field(
            alias="configOptions",
            description="The full set of configuration options and their current values.",
        ),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("config_options", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class ClientCapabilities(BaseModel):
    fs: Annotated[
        Optional[FileSystemCapabilities],
        Field(
            description="File system capabilities supported by the client.\nDetermines which file operations the agent can request.",
            validate_default=True,
        ),
    ] = FileSystemCapabilities()
    terminal: Annotated[
        Optional[bool],
        Field(description="Whether the Client support all `terminal/*` methods."),
    ] = False
    session: Annotated[
        Optional[ClientSessionCapabilities],
        Field(
            description="Session-related capabilities supported by the client.\n\nOptional. Omitted or `null` both mean the client does not advertise any\nsession-related extensions."
        ),
    ] = None
    plan: Annotated[
        Optional[PlanCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nWhether the client supports `plan_update` and `plan_removed` session updates.\n\nOptional. Omitted or `null` both mean the client does not advertise support.\nSupplying `{}` means the client can receive both update types."
        ),
    ] = None
    auth: Annotated[
        Optional[AuthCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nAuthentication capabilities supported by the client.\nDetermines which authentication method types the agent may include\nin its `InitializeResponse`.",
            validate_default=True,
        ),
    ] = {"terminal": False}
    elicitation: Annotated[
        Optional[ElicitationCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nElicitation capabilities supported by the client.\nDetermines which elicitation modes the agent may use.\n\nOptional. Omitted or `null` both mean the client does not advertise\nelicitation support."
        ),
    ] = None
    nes: Annotated[
        Optional[ClientNesCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nNES (Next Edit Suggestions) capabilities supported by the client.\n\nOptional. Omitted or `null` both mean the client does not advertise any\nNES suggestion-kind extensions."
        ),
    ] = None
    position_encodings: Annotated[
        Optional[List[str]],
        Field(
            alias="positionEncodings",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nThe position encodings supported by the client, in order of preference.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    uri: Annotated[str, Field(description="The URI of the document to suggest for.")]
    version: Annotated[int, Field(description="The version number of the document.")]
    position: Annotated[Position, Field(description="The current cursor position.")]
    selection: Annotated[Optional[Range], Field(description="The current text selection range, if any.")] = None
    trigger_kind: Annotated[
        str,
        Field(alias="triggerKind", description="What triggered this suggestion request."),
    ]
    context: Annotated[
        Optional[NesSuggestContext],
        Field(description="Context for the suggestion, included based on agent capabilities."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ClientResponseMessage(BaseModel):
    id: Annotated[
        Optional[Union[int, str]],
        Field(description="The id of the request this response answers."),
    ]
    result: Annotated[
        Union[
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
        ],
        Field(description="Method-specific response data."),
    ]


class ClientResponse(RootModel[Union[ClientResponseMessage, ClientErrorMessage]]):
    root: Annotated[
        Union[ClientResponseMessage, ClientErrorMessage],
        Field(description="A JSON-RPC response object."),
    ]


class ClientNotification(BaseModel):
    method: Annotated[str, Field(description="The notification method name.")]
    params: Annotated[
        Optional[
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
        ],
        Field(description="Method-specific notification parameters."),
    ] = None


class ToolCallUpdate(BaseModel):
    tool_call_id: Annotated[
        str,
        Field(alias="toolCallId", description="The ID of the tool call being updated."),
    ]
    kind: Annotated[Optional[ToolKind], Field(description="Update the tool kind.")] = None
    status: Annotated[Optional[ToolCallStatus], Field(description="Update the execution status.")] = None
    title: Annotated[Optional[str], Field(description="Update the human-readable title.")] = None
    content: Annotated[
        Optional[
            List[
                Annotated[
                    Union[ContentToolCallContent, FileEditToolCallContent, TerminalToolCallContent],
                    Field(discriminator="type"),
                ]
            ]
        ],
        Field(description="Replace the content collection."),
    ] = None
    locations: Annotated[
        Optional[List[ToolCallLocation]],
        Field(description="Replace the locations collection."),
    ] = None
    raw_input: Annotated[Optional[Any], Field(alias="rawInput", description="Update the raw input.")] = None
    raw_output: Annotated[Optional[Any], Field(alias="rawOutput", description="Update the raw output.")] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
    message: Annotated[
        str,
        Field(description="A human-readable message describing what input is needed."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    mode: Literal["form"]
    requested_schema: Annotated[
        ElicitationSchema,
        Field(
            alias="requestedSchema",
            description="A JSON Schema describing the form fields to present to the user.",
        ),
    ]


class CreateFormRequestElicitationRequest(ElicitationRequestScope):
    message: Annotated[
        str,
        Field(description="A human-readable message describing what input is needed."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    mode: Literal["form"]
    requested_schema: Annotated[
        ElicitationSchema,
        Field(
            alias="requestedSchema",
            description="A JSON Schema describing the form fields to present to the user.",
        ),
    ]


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
    events: Annotated[
        Optional[NesEventCapabilities],
        Field(description="Events the agent wants to receive."),
    ] = None
    context: Annotated[
        Optional[NesContextCapabilities],
        Field(description="Context the agent wants attached to each suggestion request."),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("context", "events", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class NewSessionResponse(BaseModel):
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="Unique identifier for the created session.\n\nUsed in all subsequent requests for this conversation.",
        ),
    ]
    modes: Annotated[
        Optional[SessionModeState],
        Field(
            description="Initial mode state if supported by the Agent\n\nSee protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)"
        ),
    ] = None
    config_options: Annotated[
        Optional[
            List[
                Annotated[
                    Union[SessionConfigOptionSelect, SessionConfigOptionBoolean],
                    Field(discriminator="type"),
                ]
            ]
        ],
        Field(
            alias="configOptions",
            description="Initial session configuration options if supported by the Agent.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

    @field_validator("modes", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)

    @field_validator("config_options", mode="wrap")
    @classmethod
    def _skip_invalid_items_0(cls, value: Any, handler: Any) -> Any:
        return skip_invalid_items(value, handler)


class SuggestNesResponse(BaseModel):
    suggestions: Annotated[
        List[
            Annotated[
                Union[
                    NesEditSuggestionVariant,
                    NesJumpSuggestionVariant,
                    NesRenameSuggestionVariant,
                    NesSearchAndReplaceSuggestionVariant,
                ],
                Field(discriminator="kind"),
            ]
        ],
        Field(description="The list of suggestions."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ToolCallStart(ToolCall):
    session_update: Annotated[Literal["tool_call"], Field(alias="sessionUpdate")]


class ToolCallProgress(ToolCallUpdate):
    session_update: Annotated[Literal["tool_call_update"], Field(alias="sessionUpdate")]


class ConfigOptionUpdate(_ConfigOptionUpdate):
    session_update: Annotated[Literal["config_option_update"], Field(alias="sessionUpdate")]


class InitializeRequest(BaseModel):
    protocol_version: Annotated[
        int,
        Field(
            alias="protocolVersion",
            description="The latest protocol version supported by the client.",
            ge=0,
            le=65535,
        ),
    ]
    client_capabilities: Annotated[
        Optional[ClientCapabilities],
        Field(
            alias="clientCapabilities",
            description="Capabilities supported by the client.",
            validate_default=True,
        ),
    ] = ClientCapabilities()
    client_info: Annotated[
        Optional[Implementation],
        Field(
            alias="clientInfo",
            description="Information about the Client name and version sent to the Agent.\n\nNote: in future versions of the protocol, this will be required.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    tool_call: Annotated[
        ToolCallUpdate,
        Field(
            alias="toolCall",
            description="Details about the tool call requiring permission.",
        ),
    ]
    options: Annotated[
        List[PermissionOption],
        Field(description="Available permission options for the user to choose from."),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class AgentCapabilities(BaseModel):
    load_session: Annotated[
        Optional[bool],
        Field(
            alias="loadSession",
            description="Whether the agent supports `session/load`.",
        ),
    ] = False
    prompt_capabilities: Annotated[
        Optional[PromptCapabilities],
        Field(
            alias="promptCapabilities",
            description="Prompt capabilities supported by the agent.",
            validate_default=True,
        ),
    ] = PromptCapabilities()
    mcp_capabilities: Annotated[
        Optional[McpCapabilities],
        Field(
            alias="mcpCapabilities",
            description="MCP capabilities supported by the agent.",
            validate_default=True,
        ),
    ] = McpCapabilities()
    session_capabilities: Annotated[
        Optional[SessionCapabilities],
        Field(
            alias="sessionCapabilities",
            description="Session lifecycle and prompt capabilities advertised by the agent.",
            validate_default=True,
        ),
    ] = SessionCapabilities()
    auth: Annotated[
        Optional[AgentAuthCapabilities],
        Field(
            description="Authentication-related capabilities supported by the agent.",
            validate_default=True,
        ),
    ] = {}
    providers: Annotated[
        Optional[ProvidersCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nProvider configuration capabilities supported by the agent.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports provider configuration methods."
        ),
    ] = None
    nes: Annotated[
        Optional[NesCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nNES (Next Edit Suggestions) capabilities supported by the agent.\n\nOptional. Omitted or `null` both mean the agent does not advertise support\nfor NES methods."
        ),
    ] = None
    position_encoding: Annotated[
        Optional[str],
        Field(
            alias="positionEncoding",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nThe position encoding selected by the agent from the client's supported encodings.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session this update pertains to.",
        ),
    ]
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
        Field(description="The actual update content.", discriminator="session_update"),
    ]
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ClientRequest(BaseModel):
    id: Annotated[
        Optional[Union[int, str]],
        Field(description="The request id used to correlate the matching response."),
    ]
    method: Annotated[str, Field(description="The method name to invoke.")]
    params: Annotated[
        Optional[
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
        ],
        Field(description="Method-specific request parameters."),
    ]


class AgentRequest(BaseModel):
    id: Annotated[
        Optional[Union[int, str]],
        Field(description="The request id used to correlate the matching response."),
    ]
    method: Annotated[str, Field(description="The method name to invoke.")]
    params: Annotated[
        Optional[
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
        ],
        Field(description="Method-specific request parameters."),
    ]


class InitializeResponse(BaseModel):
    protocol_version: Annotated[
        int,
        Field(
            alias="protocolVersion",
            description="The protocol version the client specified if supported by the agent,\nor the latest protocol version supported by the agent.\n\nThe client should disconnect, if it doesn't support this version.",
            ge=0,
            le=65535,
        ),
    ]
    agent_capabilities: Annotated[
        Optional[AgentCapabilities],
        Field(
            alias="agentCapabilities",
            description="Capabilities supported by the agent.",
            validate_default=True,
        ),
    ] = AgentCapabilities()
    auth_methods: Annotated[
        Optional[List[Union[EnvVarAuthMethod, TerminalAuthMethod, AuthMethodAgent]]],
        Field(
            alias="authMethods",
            description="Authentication methods supported by the agent.",
            validate_default=True,
        ),
    ] = []
    agent_info: Annotated[
        Optional[Implementation],
        Field(
            alias="agentInfo",
            description="Information about the Agent name and version sent to the Client.\n\nNote: in future versions of the protocol, this will be required.",
        ),
    ] = None
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None

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
    method: Annotated[str, Field(description="The notification method name.")]
    params: Annotated[
        Optional[
            Union[
                SessionNotification,
                CompleteElicitationNotification,
                MessageMcpNotification,
                Any,
            ]
        ],
        Field(description="Method-specific notification parameters."),
    ] = None


class AgentResponseMessage(BaseModel):
    id: Annotated[
        Optional[Union[int, str]],
        Field(description="The id of the request this response answers."),
    ]
    result: Annotated[
        Union[
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
        ],
        Field(description="Method-specific response data."),
    ]


class AgentResponse(RootModel[Union[AgentResponseMessage, AgentErrorMessage]]):
    root: Annotated[
        Union[AgentResponseMessage, AgentErrorMessage],
        Field(description="A JSON-RPC response object."),
    ]
