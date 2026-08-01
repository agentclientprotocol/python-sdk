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
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # Absolute path to the file to read.
    path: Annotated[str, Field(description="Absolute path to the file to read.")]
    # Line number to start reading from (1-based).
    line: Annotated[
        Optional[int],
        Field(description="Line number to start reading from (1-based).", ge=0),
    ] = None
    # Maximum number of lines to read.
    limit: Annotated[Optional[int], Field(description="Maximum number of lines to read.", ge=0)] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # MIME type describing the encoded media payload.
    mime_type: Annotated[
        Optional[str],
        Field(
            alias="mimeType",
            description="MIME type describing the encoded media payload.",
        ),
    ] = None
    # Text payload carried by this content block.
    text: Annotated[str, Field(description="Text payload carried by this content block.")]
    # URI associated with this resource or media payload.
    uri: Annotated[str, Field(description="URI associated with this resource or media payload.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Base64-encoded bytes for a binary resource payload.
    blob: Annotated[str, Field(description="Base64-encoded bytes for a binary resource payload.")]
    # MIME type describing the encoded media payload.
    mime_type: Annotated[
        Optional[str],
        Field(
            alias="mimeType",
            description="MIME type describing the encoded media payload.",
        ),
    ] = None
    # URI associated with this resource or media payload.
    uri: Annotated[str, Field(description="URI associated with this resource or media payload.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The absolute file path being modified.
    path: Annotated[str, Field(description="The absolute file path being modified.")]
    # The original content (None for new files).
    old_text: Annotated[
        Optional[str],
        Field(alias="oldText", description="The original content (None for new files)."),
    ] = None
    # The new content after modification.
    new_text: Annotated[str, Field(alias="newText", description="The new content after modification.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Identifier of the terminal instance to embed in the content stream.
    terminal_id: Annotated[
        str,
        Field(
            alias="terminalId",
            description="Identifier of the terminal instance to embed in the content stream.",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ToolCallLocation(BaseModel):
    # The absolute file path being accessed or modified.
    path: Annotated[str, Field(description="The absolute file path being accessed or modified.")]
    # Optional line number within the file.
    line: Annotated[Optional[int], Field(description="Optional line number within the file.", ge=0)] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The name of the environment variable.
    name: Annotated[str, Field(description="The name of the environment variable.")]
    # The value to set for the environment variable.
    value: Annotated[str, Field(description="The value to set for the environment variable.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class TerminalOutputRequest(BaseModel):
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # The ID of the terminal to get output from.
    terminal_id: Annotated[
        str,
        Field(alias="terminalId", description="The ID of the terminal to get output from."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ReleaseTerminalRequest(BaseModel):
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # The ID of the terminal to release.
    terminal_id: Annotated[str, Field(alias="terminalId", description="The ID of the terminal to release.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class WaitForTerminalExitRequest(BaseModel):
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # The ID of the terminal to wait for.
    terminal_id: Annotated[
        str,
        Field(alias="terminalId", description="The ID of the terminal to wait for."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class KillTerminalRequest(BaseModel):
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # The ID of the terminal to kill.
    terminal_id: Annotated[str, Field(alias="terminalId", description="The ID of the terminal to kill.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # A human-readable message describing what input is needed.
    message: Annotated[
        str,
        Field(description="A human-readable message describing what input is needed."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    # Custom or future elicitation mode.
    #
    # Values beginning with `_` are reserved for implementation-specific
    # extensions. Unknown values that do not begin with `_` are reserved for
    # future ACP variants.
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
    # The session this elicitation is tied to.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session this elicitation is tied to."),
    ]
    # Optional tool call within the session.
    tool_call_id: Annotated[
        Optional[str],
        Field(alias="toolCallId", description="Optional tool call within the session."),
    ] = None

    @field_validator("tool_call_id", mode="wrap")
    @classmethod
    def _salvage_on_error_0(cls, value: Any, handler: Any) -> Any:
        return salvage_on_error(value, handler, lambda: None)


class ElicitationRequestScope(BaseModel):
    # The request this elicitation is tied to.
    request_id: Annotated[
        Optional[Union[int, str]],
        Field(alias="requestId", description="The request this elicitation is tied to."),
    ]


class ElicitationOtherPropertySchema(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    # Custom or future elicitation property schema type.
    #
    # Values beginning with `_` are reserved for implementation-specific
    # extensions. Unknown values that do not begin with `_` are reserved for
    # future ACP variants.
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
    # The constant value for this option.
    const: Annotated[str, Field(description="The constant value for this option.")]
    # Human-readable title for this option.
    title: Annotated[str, Field(description="Human-readable title for this option.")]
    # Human-readable description.
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Optional title for the property.
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None
    # Human-readable description.
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    # Minimum string length.
    min_length: Annotated[
        Optional[int],
        Field(alias="minLength", description="Minimum string length.", ge=0),
    ] = None
    # Maximum string length.
    max_length: Annotated[
        Optional[int],
        Field(alias="maxLength", description="Maximum string length.", ge=0),
    ] = None
    # Pattern the string must match.
    pattern: Annotated[Optional[str], Field(description="Pattern the string must match.")] = None
    # String format.
    format: Annotated[Optional[str], Field(description="String format.")] = None
    # Default value.
    default: Annotated[Optional[str], Field(description="Default value.")] = None
    # Enum values for untitled single-select enums.
    enum: Annotated[
        Optional[List[str]],
        Field(description="Enum values for untitled single-select enums."),
    ] = None
    # Titled enum options for titled single-select enums.
    one_of: Annotated[
        Optional[List[EnumOption]],
        Field(
            alias="oneOf",
            description="Titled enum options for titled single-select enums.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Optional title for the property.
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None
    # Human-readable description.
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    # Minimum value (inclusive).
    minimum: Annotated[Optional[float], Field(description="Minimum value (inclusive).")] = None
    # Maximum value (inclusive).
    maximum: Annotated[Optional[float], Field(description="Maximum value (inclusive).")] = None
    # Default value.
    default: Annotated[Optional[float], Field(description="Default value.")] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Optional title for the property.
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None
    # Human-readable description.
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    # Minimum value (inclusive).
    minimum: Annotated[Optional[int], Field(description="Minimum value (inclusive).")] = None
    # Maximum value (inclusive).
    maximum: Annotated[Optional[int], Field(description="Maximum value (inclusive).")] = None
    # Default value.
    default: Annotated[Optional[int], Field(description="Default value.")] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Optional title for the property.
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None
    # Human-readable description.
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    # Default value.
    default: Annotated[Optional[bool], Field(description="Default value.")] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Custom or future multi-select item type.
    #
    # Values beginning with `_` are reserved for implementation-specific
    # extensions. Unknown values that do not begin with `_` are reserved for
    # future ACP variants.
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
    # Allowed enum values.
    enum: Annotated[List[str], Field(description="Allowed enum values.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class TitledMultiSelectItems(BaseModel):
    # Titled enum options.
    any_of: Annotated[List[EnumOption], Field(alias="anyOf", description="Titled enum options.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ElicitationUrlSessionMode(ElicitationSessionScope):
    # The unique identifier for this elicitation.
    elicitation_id: Annotated[
        str,
        Field(
            alias="elicitationId",
            description="The unique identifier for this elicitation.",
        ),
    ]
    # The URL to direct the user to.
    url: Annotated[AnyUrl, Field(description="The URL to direct the user to.")]


class ElicitationUrlRequestMode(ElicitationRequestScope):
    # The unique identifier for this elicitation.
    elicitation_id: Annotated[
        str,
        Field(
            alias="elicitationId",
            description="The unique identifier for this elicitation.",
        ),
    ]
    # The URL to direct the user to.
    url: Annotated[AnyUrl, Field(description="The URL to direct the user to.")]


class ElicitationUrlMode(RootModel[Union[ElicitationUrlSessionMode, ElicitationUrlRequestMode]]):
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # URL-based elicitation mode where the client directs the user to a URL.
    root: Annotated[
        Union[ElicitationUrlSessionMode, ElicitationUrlRequestMode],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nURL-based elicitation mode where the client directs the user to a URL."
        ),
    ]


class DisconnectMcpRequest(BaseModel):
    # The MCP-over-ACP connection to close.
    connection_id: Annotated[
        str,
        Field(alias="connectionId", description="The MCP-over-ACP connection to close."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class PromptCapabilities(BaseModel):
    # Agent supports [`ContentBlock::Image`].
    image: Annotated[Optional[bool], Field(description="Agent supports [`ContentBlock::Image`].")] = False
    # Agent supports [`ContentBlock::Audio`].
    audio: Annotated[Optional[bool], Field(description="Agent supports [`ContentBlock::Audio`].")] = False
    # Agent supports embedded context in `session/prompt` requests.
    #
    # When enabled, the Client is allowed to include [`ContentBlock::Resource`]
    # in prompt requests for pieces of context that are referenced in the message.
    embedded_context: Annotated[
        Optional[bool],
        Field(
            alias="embeddedContext",
            description="Agent supports embedded context in `session/prompt` requests.\n\nWhen enabled, the Client is allowed to include [`ContentBlock::Resource`]\nin prompt requests for pieces of context that are referenced in the message.",
        ),
    ] = False
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Agent supports [`McpServer::Http`].
    http: Annotated[Optional[bool], Field(description="Agent supports [`McpServer::Http`].")] = False
    # Agent supports [`McpServer::Sse`].
    sse: Annotated[Optional[bool], Field(description="Agent supports [`McpServer::Sse`].")] = False
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Agent supports [`McpServer::Acp`].
    acp: Annotated[
        Optional[bool],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nAgent supports [`McpServer::Acp`]."
        ),
    ] = False
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionDeleteCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionAdditionalDirectoriesCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionForkCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionResumeCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionCloseCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class LogoutCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ProvidersCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesDocumentDidOpenCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesDocumentDidCloseCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesDocumentDidSaveCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesDocumentDidFocusCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesRecentFilesCapabilities(BaseModel):
    # Maximum number of recent files the agent can use.
    max_count: Annotated[
        Optional[int],
        Field(
            alias="maxCount",
            description="Maximum number of recent files the agent can use.",
            ge=0,
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesEditHistoryCapabilities(BaseModel):
    # Maximum number of edit history entries the agent can use.
    max_count: Annotated[
        Optional[int],
        Field(
            alias="maxCount",
            description="Maximum number of edit history entries the agent can use.",
            ge=0,
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Maximum number of user actions the agent can use.
    max_count: Annotated[
        Optional[int],
        Field(
            alias="maxCount",
            description="Maximum number of user actions the agent can use.",
            ge=0,
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesDiagnosticsCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class AuthEnvVar(BaseModel):
    # The environment variable name (e.g. `"OPENAI_API_KEY"`).
    name: Annotated[
        str,
        Field(description='The environment variable name (e.g. `"OPENAI_API_KEY"`).'),
    ]
    # Human-readable label for this variable, displayed in client UI.
    label: Annotated[
        Optional[str],
        Field(description="Human-readable label for this variable, displayed in client UI."),
    ] = None
    # Whether this value is a secret (e.g. API key, token).
    # Clients should use a password-style input for secret vars.
    #
    # Defaults to `true`.
    secret: Annotated[
        Optional[bool],
        Field(
            description="Whether this value is a secret (e.g. API key, token).\nClients should use a password-style input for secret vars.\n\nDefaults to `true`."
        ),
    ] = True
    # Whether this variable is optional.
    #
    # Defaults to `false`.
    optional: Annotated[
        Optional[bool],
        Field(description="Whether this variable is optional.\n\nDefaults to `false`."),
    ] = False
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Unique identifier for this authentication method.
    id: Annotated[str, Field(description="Unique identifier for this authentication method.")]
    # Human-readable name of the authentication method.
    name: Annotated[str, Field(description="Human-readable name of the authentication method.")]
    # Optional description providing more details about this authentication method.
    description: Annotated[
        Optional[str],
        Field(description="Optional description providing more details about this authentication method."),
    ] = None
    # The environment variables the client should set.
    vars: Annotated[
        List[AuthEnvVar],
        Field(description="The environment variables the client should set."),
    ]
    # Optional link to a page where the user can obtain their credentials.
    link: Annotated[
        Optional[str],
        Field(description="Optional link to a page where the user can obtain their credentials."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Unique identifier for this authentication method.
    id: Annotated[str, Field(description="Unique identifier for this authentication method.")]
    # Human-readable name of the authentication method.
    name: Annotated[str, Field(description="Human-readable name of the authentication method.")]
    # Optional description providing more details about this authentication method.
    description: Annotated[
        Optional[str],
        Field(description="Optional description providing more details about this authentication method."),
    ] = None
    # Additional arguments to pass when running the agent binary for terminal auth.
    args: Annotated[
        Optional[List[str]],
        Field(description="Additional arguments to pass when running the agent binary for terminal auth."),
    ] = None
    # Additional environment variables to set when running the agent binary for terminal auth.
    env: Annotated[
        Optional[Dict[str, str]],
        Field(description="Additional environment variables to set when running the agent binary for terminal auth."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Unique identifier for this authentication method.
    id: Annotated[str, Field(description="Unique identifier for this authentication method.")]
    # Human-readable name of the authentication method.
    name: Annotated[str, Field(description="Human-readable name of the authentication method.")]
    # Optional description providing more details about this authentication method.
    description: Annotated[
        Optional[str],
        Field(description="Optional description providing more details about this authentication method."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Intended for programmatic or logical use, but can be used as a display
    # name fallback if title isn’t present.
    name: Annotated[
        str,
        Field(
            description="Intended for programmatic or logical use, but can be used as a display\nname fallback if title isn’t present."
        ),
    ]
    # Intended for UI and end-user contexts — optimized to be human-readable
    # and easily understood.
    #
    # If not provided, the name should be used for display.
    title: Annotated[
        Optional[str],
        Field(
            description="Intended for UI and end-user contexts — optimized to be human-readable\nand easily understood.\n\nIf not provided, the name should be used for display."
        ),
    ] = None
    # Version of the implementation. Can be displayed to the user or used
    # for debugging or metrics purposes. (e.g. "1.0.0").
    version: Annotated[
        str,
        Field(
            description='Version of the implementation. Can be displayed to the user or used\nfor debugging or metrics purposes. (e.g. "1.0.0").'
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ProviderCurrentConfig(BaseModel):
    # Protocol currently used by this provider.
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
    # Base URL currently used by this provider.
    base_url: Annotated[
        str,
        Field(alias="baseUrl", description="Base URL currently used by this provider."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SetProviderResponse(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DisableProviderResponse(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class LogoutResponse(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionMode(BaseModel):
    # Stable identifier used to refer to this protocol object in later messages.
    id: Annotated[
        str,
        Field(description="Stable identifier used to refer to this protocol object in later messages."),
    ]
    # Human-readable name shown for this protocol object.
    name: Annotated[str, Field(description="Human-readable name shown for this protocol object.")]
    # Optional human-readable details shown with this protocol object.
    description: Annotated[
        Optional[str],
        Field(description="Optional human-readable details shown with this protocol object."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Unique identifier for this option value.
    value: Annotated[str, Field(description="Unique identifier for this option value.")]
    # Human-readable label for this option value.
    name: Annotated[str, Field(description="Human-readable label for this option value.")]
    # Optional description for this option value.
    description: Annotated[Optional[str], Field(description="Optional description for this option value.")] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The current value of the boolean option.
    current_value: Annotated[
        bool,
        Field(alias="currentValue", description="The current value of the boolean option."),
    ]


class SessionInfo(BaseModel):
    # Unique identifier for the session
    session_id: Annotated[str, Field(alias="sessionId", description="Unique identifier for the session")]
    # The working directory for this session. Must be an absolute path.
    cwd: Annotated[
        str,
        Field(description="The working directory for this session. Must be an absolute path."),
    ]
    # Additional workspace roots reported for this session. Each path must be absolute.
    #
    # When present, this is the complete ordered additional-root list reported
    # by the Agent. Omitted and empty values are equivalent: the response
    # reports no additional roots.
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="Additional workspace roots reported for this session. Each path must be absolute.\n\nWhen present, this is the complete ordered additional-root list reported\nby the Agent. Omitted and empty values are equivalent: the response\nreports no additional roots.",
        ),
    ] = None
    # Human-readable title for the session
    title: Annotated[Optional[str], Field(description="Human-readable title for the session")] = None
    # ISO 8601 timestamp of last activity
    updated_at: Annotated[
        Optional[str],
        Field(alias="updatedAt", description="ISO 8601 timestamp of last activity"),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CloseSessionResponse(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SetSessionModeResponse(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class Usage(BaseModel):
    # Sum of all token types across session.
    total_tokens: Annotated[
        int,
        Field(
            alias="totalTokens",
            description="Sum of all token types across session.",
            ge=0,
        ),
    ]
    # Total input tokens across all turns.
    input_tokens: Annotated[
        int,
        Field(
            alias="inputTokens",
            description="Total input tokens across all turns.",
            ge=0,
        ),
    ]
    # Total output tokens across all turns.
    output_tokens: Annotated[
        int,
        Field(
            alias="outputTokens",
            description="Total output tokens across all turns.",
            ge=0,
        ),
    ]
    # Total thought/reasoning tokens
    thought_tokens: Annotated[
        Optional[int],
        Field(alias="thoughtTokens", description="Total thought/reasoning tokens", ge=0),
    ] = None
    # Total cache read tokens.
    cached_read_tokens: Annotated[
        Optional[int],
        Field(alias="cachedReadTokens", description="Total cache read tokens.", ge=0),
    ] = None
    # Total cache write tokens.
    cached_write_tokens: Annotated[
        Optional[int],
        Field(alias="cachedWriteTokens", description="Total cache write tokens.", ge=0),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The session ID for the newly started NES session.
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The session ID for the newly started NES session.",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class Position(BaseModel):
    # Zero-based line number.
    line: Annotated[int, Field(description="Zero-based line number.", ge=0)]
    # Zero-based character offset (encoding-dependent).
    character: Annotated[
        int,
        Field(description="Zero-based character offset (encoding-dependent).", ge=0),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesJumpSuggestion(BaseModel):
    # Unique identifier for accept/reject tracking.
    id: Annotated[str, Field(description="Unique identifier for accept/reject tracking.")]
    # The file to navigate to.
    uri: Annotated[str, Field(description="The file to navigate to.")]
    # The target position within the file.
    position: Annotated[Position, Field(description="The target position within the file.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesRenameSuggestion(BaseModel):
    # Unique identifier for accept/reject tracking.
    id: Annotated[str, Field(description="Unique identifier for accept/reject tracking.")]
    # The file URI containing the symbol.
    uri: Annotated[str, Field(description="The file URI containing the symbol.")]
    # The position of the symbol to rename.
    position: Annotated[Position, Field(description="The position of the symbol to rename.")]
    # The new name for the symbol.
    new_name: Annotated[str, Field(alias="newName", description="The new name for the symbol.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesSearchAndReplaceSuggestion(BaseModel):
    # Unique identifier for accept/reject tracking.
    id: Annotated[str, Field(description="Unique identifier for accept/reject tracking.")]
    # The file URI to search within.
    uri: Annotated[str, Field(description="The file URI to search within.")]
    # The text or pattern to find.
    search: Annotated[str, Field(description="The text or pattern to find.")]
    # The replacement text.
    replace: Annotated[str, Field(description="The replacement text.")]
    # Whether `search` is a regular expression. Defaults to `false`.
    is_regex: Annotated[
        Optional[bool],
        Field(
            alias="isRegex",
            description="Whether `search` is a regular expression. Defaults to `false`.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CloseNesResponse(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class PlanFile(BaseModel):
    # The plan ID to update.
    plan_id: Annotated[str, Field(alias="planId", description="The plan ID to update.")]
    # The URI of the file containing the plan.
    uri: Annotated[str, Field(description="The URI of the file containing the plan.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class PlanMarkdown(BaseModel):
    # The plan ID to update.
    plan_id: Annotated[str, Field(alias="planId", description="The plan ID to update.")]
    # Markdown content for the plan.
    content: Annotated[str, Field(description="Markdown content for the plan.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class PlanRemoved(BaseModel):
    # The plan ID to remove.
    plan_id: Annotated[str, Field(alias="planId", description="The plan ID to remove.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class UnstructuredCommandInput(BaseModel):
    # A hint to display when the input hasn't been provided yet
    hint: Annotated[
        str,
        Field(description="A hint to display when the input hasn't been provided yet"),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class _CurrentModeUpdate(BaseModel):
    # The ID of the current mode
    current_mode_id: Annotated[str, Field(alias="currentModeId", description="The ID of the current mode")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class _SessionInfoUpdate(BaseModel):
    # Human-readable title for the session. Set to null to clear.
    title: Annotated[
        Optional[str],
        Field(description="Human-readable title for the session. Set to null to clear."),
    ] = None
    # ISO 8601 timestamp of last activity. Set to null to clear.
    updated_at: Annotated[
        Optional[str],
        Field(
            alias="updatedAt",
            description="ISO 8601 timestamp of last activity. Set to null to clear.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Total cumulative cost for session.
    amount: Annotated[float, Field(description="Total cumulative cost for session.")]
    # ISO 4217 currency code (e.g., "USD", "EUR").
    currency: Annotated[str, Field(description='ISO 4217 currency code (e.g., "USD", "EUR").')]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class _UsageUpdate(BaseModel):
    # Tokens currently in context.
    used: Annotated[int, Field(description="Tokens currently in context.", ge=0)]
    # Total context window size in tokens.
    size: Annotated[int, Field(description="Total context window size in tokens.", ge=0)]
    # Cumulative session cost (optional).
    cost: Annotated[Optional[Cost], Field(description="Cumulative session cost (optional).")] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The ID of the elicitation that completed.
    elicitation_id: Annotated[
        str,
        Field(
            alias="elicitationId",
            description="The ID of the elicitation that completed.",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class MessageMcpNotification(BaseModel):
    # The MCP-over-ACP connection this message is sent on.
    connection_id: Annotated[
        str,
        Field(
            alias="connectionId",
            description="The MCP-over-ACP connection this message is sent on.",
        ),
    ]
    # The inner MCP method name.
    method: Annotated[str, Field(description="The inner MCP method name.")]
    # Optional inner MCP params.
    #
    # If omitted or set to `null`, the inner MCP message has no params.
    params: Annotated[
        Optional[Dict[str, Any]],
        Field(
            description="Optional inner MCP params.\n\nIf omitted or set to `null`, the inner MCP message has no params."
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Whether the Client supports `fs/read_text_file` requests.
    read_text_file: Annotated[
        Optional[bool],
        Field(
            alias="readTextFile",
            description="Whether the Client supports `fs/read_text_file` requests.",
        ),
    ] = False
    # Whether the Client supports `fs/write_text_file` requests.
    write_text_file: Annotated[
        Optional[bool],
        Field(
            alias="writeTextFile",
            description="Whether the Client supports `fs/write_text_file` requests.",
        ),
    ] = False
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class PlanCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class AuthCapabilities(BaseModel):
    # Whether the client supports `terminal` authentication methods.
    #
    # When `true`, the agent may include `terminal` entries in its authentication methods.
    terminal: Annotated[
        Optional[bool],
        Field(
            description="Whether the client supports `terminal` authentication methods.\n\nWhen `true`, the agent may include `terminal` entries in its authentication methods."
        ),
    ] = False
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ElicitationUrlCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesJumpCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesRenameCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesSearchAndReplaceCapabilities(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class AuthenticateRequest(BaseModel):
    # The ID of the authentication method to use.
    # Must be one of the methods advertised in the initialize response.
    method_id: Annotated[
        str,
        Field(
            alias="methodId",
            description="The ID of the authentication method to use.\nMust be one of the methods advertised in the initialize response.",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ListProvidersRequest(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SetProviderRequest(BaseModel):
    # Provider ID to configure.
    provider_id: Annotated[str, Field(alias="providerId", description="Provider ID to configure.")]
    # Protocol type for this provider.
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
    # Base URL for requests sent through this provider.
    base_url: Annotated[
        str,
        Field(
            alias="baseUrl",
            description="Base URL for requests sent through this provider.",
        ),
    ]
    # Full headers map for this provider.
    # May include authorization, routing, or other integration-specific headers.
    headers: Annotated[
        Optional[Dict[str, str]],
        Field(
            description="Full headers map for this provider.\nMay include authorization, routing, or other integration-specific headers."
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DisableProviderRequest(BaseModel):
    # Provider ID to disable.
    provider_id: Annotated[str, Field(alias="providerId", description="Provider ID to disable.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class LogoutRequest(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class HttpHeader(BaseModel):
    # The name of the HTTP header.
    name: Annotated[str, Field(description="The name of the HTTP header.")]
    # The value to set for the HTTP header.
    value: Annotated[str, Field(description="The value to set for the HTTP header.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class McpServerHttp(BaseModel):
    # Human-readable name identifying this MCP server.
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]
    # URL to the MCP server.
    url: Annotated[str, Field(description="URL to the MCP server.")]
    # HTTP headers to set when making requests to the MCP server.
    headers: Annotated[
        List[HttpHeader],
        Field(description="HTTP headers to set when making requests to the MCP server."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class McpServerSse(BaseModel):
    # Human-readable name identifying this MCP server.
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]
    # URL to the MCP server.
    url: Annotated[str, Field(description="URL to the MCP server.")]
    # HTTP headers to set when making requests to the MCP server.
    headers: Annotated[
        List[HttpHeader],
        Field(description="HTTP headers to set when making requests to the MCP server."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class McpServerAcp(BaseModel):
    # Human-readable name identifying this MCP server.
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]
    # Unique identifier for this MCP server, generated by the component providing it.
    #
    # Providers MUST NOT reuse an ID for multiple ACP-transport MCP servers that are visible
    # on the same ACP connection.
    server_id: Annotated[
        str,
        Field(
            alias="serverId",
            description="Unique identifier for this MCP server, generated by the component providing it.\n\nProviders MUST NOT reuse an ID for multiple ACP-transport MCP servers that are visible\non the same ACP connection.",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class McpServerStdio(BaseModel):
    # Human-readable name identifying this MCP server.
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]
    # Absolute path to the MCP server executable.
    command: Annotated[str, Field(description="Absolute path to the MCP server executable.")]
    # Command-line arguments to pass to the MCP server.
    args: Annotated[
        List[str],
        Field(description="Command-line arguments to pass to the MCP server."),
    ]
    # Environment variables to set when launching the MCP server.
    env: Annotated[
        List[EnvVariable],
        Field(description="Environment variables to set when launching the MCP server."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ListSessionsRequest(BaseModel):
    # Filter sessions by working directory. Must be an absolute path.
    cwd: Annotated[
        Optional[str],
        Field(description="Filter sessions by working directory. Must be an absolute path."),
    ] = None
    # Opaque cursor token from a previous response's nextCursor field for cursor-based pagination
    cursor: Annotated[
        Optional[str],
        Field(
            description="Opaque cursor token from a previous response's nextCursor field for cursor-based pagination"
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DeleteSessionRequest(BaseModel):
    # The ID of the session to delete.
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to delete.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CloseSessionRequest(BaseModel):
    # The ID of the session to close.
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to close.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SetSessionModeRequest(BaseModel):
    # The ID of the session to set the mode for.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The ID of the session to set the mode for."),
    ]
    # The ID of the mode to set.
    mode_id: Annotated[str, Field(alias="modeId", description="The ID of the mode to set.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SetSessionConfigOptionBooleanRequest(BaseModel):
    # The ID of the session to set the configuration option for.
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session to set the configuration option for.",
        ),
    ]
    # The ID of the configuration option to set.
    config_id: Annotated[
        str,
        Field(alias="configId", description="The ID of the configuration option to set."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    # The boolean value.
    value: Annotated[bool, Field(description="The boolean value.")]
    type: Literal["boolean"]


class SetSessionConfigOptionSelectRequest(BaseModel):
    # The ID of the session to set the configuration option for.
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session to set the configuration option for.",
        ),
    ]
    # The ID of the configuration option to set.
    config_id: Annotated[
        str,
        Field(alias="configId", description="The ID of the configuration option to set."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    # The value ID.
    value: Annotated[str, Field(description="The value ID.")]


class WorkspaceFolder(BaseModel):
    # The URI of the folder.
    uri: Annotated[str, Field(description="The URI of the folder.")]
    # The display name of the folder.
    name: Annotated[str, Field(description="The display name of the folder.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesRepository(BaseModel):
    # The repository name.
    name: Annotated[str, Field(description="The repository name.")]
    # The repository owner.
    owner: Annotated[str, Field(description="The repository owner.")]
    # The remote URL of the repository.
    remote_url: Annotated[str, Field(alias="remoteUrl", description="The remote URL of the repository.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesRecentFile(BaseModel):
    # The URI of the file.
    uri: Annotated[str, Field(description="The URI of the file.")]
    # The language identifier.
    language_id: Annotated[str, Field(alias="languageId", description="The language identifier.")]
    # The full text content of the file.
    text: Annotated[str, Field(description="The full text content of the file.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesExcerpt(BaseModel):
    # The start line of the excerpt (zero-based).
    start_line: Annotated[
        int,
        Field(
            alias="startLine",
            description="The start line of the excerpt (zero-based).",
            ge=0,
        ),
    ]
    # The end line of the excerpt (zero-based).
    end_line: Annotated[
        int,
        Field(
            alias="endLine",
            description="The end line of the excerpt (zero-based).",
            ge=0,
        ),
    ]
    # The text content of the excerpt.
    text: Annotated[str, Field(description="The text content of the excerpt.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesEditHistoryEntry(BaseModel):
    # The URI of the edited file.
    uri: Annotated[str, Field(description="The URI of the edited file.")]
    # A diff representing the edit.
    diff: Annotated[str, Field(description="A diff representing the edit.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesUserAction(BaseModel):
    # The kind of action (e.g., "insertChar", "cursorMovement").
    action: Annotated[
        str,
        Field(description='The kind of action (e.g., "insertChar", "cursorMovement").'),
    ]
    # The URI of the file where the action occurred.
    uri: Annotated[str, Field(description="The URI of the file where the action occurred.")]
    # The position where the action occurred.
    position: Annotated[Position, Field(description="The position where the action occurred.")]
    # Timestamp in milliseconds since epoch.
    timestamp_ms: Annotated[
        int,
        Field(
            alias="timestampMs",
            description="Timestamp in milliseconds since epoch.",
            ge=0,
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CloseNesRequest(BaseModel):
    # The ID of the NES session to close.
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the NES session to close.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class WriteTextFileResponse(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ReadTextFileResponse(BaseModel):
    # Content payload returned by this response.
    content: Annotated[str, Field(description="Content payload returned by this response.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The ID of the option the user selected.
    option_id: Annotated[
        str,
        Field(alias="optionId", description="The ID of the option the user selected."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CreateTerminalResponse(BaseModel):
    # The unique identifier for the created terminal.
    terminal_id: Annotated[
        str,
        Field(
            alias="terminalId",
            description="The unique identifier for the created terminal.",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class TerminalExitStatus(BaseModel):
    # The process exit code (may be null if terminated by signal).
    exit_code: Annotated[
        Optional[int],
        Field(
            alias="exitCode",
            description="The process exit code (may be null if terminated by signal).",
            ge=0,
        ),
    ] = None
    # The signal that terminated the process (may be null if exited normally).
    signal: Annotated[
        Optional[str],
        Field(description="The signal that terminated the process (may be null if exited normally)."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class WaitForTerminalExitResponse(BaseModel):
    # The process exit code (may be null if terminated by signal).
    exit_code: Annotated[
        Optional[int],
        Field(
            alias="exitCode",
            description="The process exit code (may be null if terminated by signal).",
            ge=0,
        ),
    ] = None
    # The signal that terminated the process (may be null if exited normally).
    signal: Annotated[
        Optional[str],
        Field(description="The signal that terminated the process (may be null if exited normally)."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DeclineElicitationResponse(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    action: Literal["decline"]


class CancelElicitationResponse(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    # Custom or future elicitation action.
    #
    # Values beginning with `_` are reserved for implementation-specific
    # extensions. Unknown values that do not begin with `_` are reserved for
    # future ACP variants.
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
    # Allowed wire representations for [`ElicitationContentValue`].
    root: Annotated[
        Union[str, int, float, bool, List[str]],
        Field(description="Allowed wire representations for [`ElicitationContentValue`]."),
    ]


class ElicitationAcceptAction(BaseModel):
    # The user-provided content, if any, as an object matching the requested schema.
    content: Annotated[
        Optional[Dict[str, Any]],
        Field(description="The user-provided content, if any, as an object matching the requested schema."),
    ] = None


class ConnectMcpResponse(BaseModel):
    # The unique identifier for this MCP-over-ACP connection.
    connection_id: Annotated[
        str,
        Field(
            alias="connectionId",
            description="The unique identifier for this MCP-over-ACP connection.",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DisconnectMcpResponse(BaseModel):
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CancelNotification(BaseModel):
    # The ID of the session to cancel operations for.
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session to cancel operations for.",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DidOpenDocumentNotification(BaseModel):
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    # The URI of the opened document.
    uri: Annotated[str, Field(description="The URI of the opened document.")]
    # The language identifier of the document (e.g., "rust", "python").
    language_id: Annotated[
        str,
        Field(
            alias="languageId",
            description='The language identifier of the document (e.g., "rust", "python").',
        ),
    ]
    # The version number of the document.
    version: Annotated[int, Field(description="The version number of the document.")]
    # The full text content of the document.
    text: Annotated[str, Field(description="The full text content of the document.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DidCloseDocumentNotification(BaseModel):
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    # The URI of the closed document.
    uri: Annotated[str, Field(description="The URI of the closed document.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DidSaveDocumentNotification(BaseModel):
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    # The URI of the saved document.
    uri: Annotated[str, Field(description="The URI of the saved document.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class AcceptNesNotification(BaseModel):
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    # The ID of the accepted suggestion.
    id: Annotated[str, Field(description="The ID of the accepted suggestion.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CancelRequestNotification(BaseModel):
    # The ID of the request to cancel.
    request_id: Annotated[
        Optional[Union[int, str]],
        Field(alias="requestId", description="The ID of the request to cancel."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ]


class WriteTextFileRequest(BaseModel):
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # Absolute path to the file to write.
    path: Annotated[str, Field(description="Absolute path to the file to write.")]
    # The text content to write to the file.
    content: Annotated[str, Field(description="The text content to write to the file.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Intended recipients for this content, such as the user or assistant.
    audience: Annotated[
        Optional[List[str]],
        Field(description="Intended recipients for this content, such as the user or assistant."),
    ] = None
    # Timestamp indicating when the underlying resource was last modified.
    last_modified: Annotated[
        Optional[str],
        Field(
            alias="lastModified",
            description="Timestamp indicating when the underlying resource was last modified.",
        ),
    ] = None
    # Relative importance of this content when clients choose what to surface.
    priority: Annotated[
        Optional[float],
        Field(description="Relative importance of this content when clients choose what to surface."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Optional annotations that help clients decide how to display or route this content.
    annotations: Annotated[
        Optional[Annotations],
        Field(description="Optional annotations that help clients decide how to display or route this content."),
    ] = None
    # Text payload carried by this content block.
    text: Annotated[str, Field(description="Text payload carried by this content block.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Optional annotations that help clients decide how to display or route this content.
    annotations: Annotated[
        Optional[Annotations],
        Field(description="Optional annotations that help clients decide how to display or route this content."),
    ] = None
    # Base64-encoded media payload.
    data: Annotated[str, Field(description="Base64-encoded media payload.")]
    # MIME type describing the encoded media payload.
    mime_type: Annotated[
        str,
        Field(
            alias="mimeType",
            description="MIME type describing the encoded media payload.",
        ),
    ]
    # URI associated with this resource or media payload.
    uri: Annotated[
        Optional[str],
        Field(description="URI associated with this resource or media payload."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Optional annotations that help clients decide how to display or route this content.
    annotations: Annotated[
        Optional[Annotations],
        Field(description="Optional annotations that help clients decide how to display or route this content."),
    ] = None
    # Base64-encoded media payload.
    data: Annotated[str, Field(description="Base64-encoded media payload.")]
    # MIME type describing the encoded media payload.
    mime_type: Annotated[
        str,
        Field(
            alias="mimeType",
            description="MIME type describing the encoded media payload.",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Optional annotations that help clients decide how to display or route this content.
    annotations: Annotated[
        Optional[Annotations],
        Field(description="Optional annotations that help clients decide how to display or route this content."),
    ] = None
    # Optional human-readable details shown with this protocol object.
    description: Annotated[
        Optional[str],
        Field(description="Optional human-readable details shown with this protocol object."),
    ] = None
    # MIME type describing the encoded media payload.
    mime_type: Annotated[
        Optional[str],
        Field(
            alias="mimeType",
            description="MIME type describing the encoded media payload.",
        ),
    ] = None
    # Human-readable name shown for this protocol object.
    name: Annotated[str, Field(description="Human-readable name shown for this protocol object.")]
    # Optional size of the linked resource in bytes, if known.
    size: Annotated[
        Optional[int],
        Field(description="Optional size of the linked resource in bytes, if known."),
    ] = None
    # Optional display title for end-user UI.
    title: Annotated[Optional[str], Field(description="Optional display title for end-user UI.")] = None
    # URI associated with this resource or media payload.
    uri: Annotated[str, Field(description="URI associated with this resource or media payload.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Optional annotations that help clients decide how to display or route this content.
    annotations: Annotated[
        Optional[Annotations],
        Field(description="Optional annotations that help clients decide how to display or route this content."),
    ] = None
    # Embedded resource payload, either text or binary data.
    resource: Annotated[
        Union[TextResourceContents, BlobResourceContents],
        Field(description="Embedded resource payload, either text or binary data."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Unique identifier for this permission option.
    option_id: Annotated[
        str,
        Field(
            alias="optionId",
            description="Unique identifier for this permission option.",
        ),
    ]
    # Human-readable label to display to the user.
    name: Annotated[str, Field(description="Human-readable label to display to the user.")]
    # Hint about the nature of this permission option.
    kind: Annotated[PermissionOptionKind, Field(description="Hint about the nature of this permission option.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class CreateTerminalRequest(BaseModel):
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # The command to execute.
    command: Annotated[str, Field(description="The command to execute.")]
    # Array of command arguments.
    args: Annotated[Optional[List[str]], Field(description="Array of command arguments.")] = None
    # Environment variables for the command.
    env: Annotated[
        Optional[List[EnvVariable]],
        Field(description="Environment variables for the command."),
    ] = None
    # Working directory for the command. Must be an absolute path.
    cwd: Annotated[
        Optional[str],
        Field(description="Working directory for the command. Must be an absolute path."),
    ] = None
    # Maximum number of output bytes to retain.
    #
    # When the limit is exceeded, the Client truncates from the beginning of the output
    # to stay within the limit.
    #
    # The Client MUST ensure truncation happens at a character boundary to maintain valid
    # string output, even if this means the retained output is slightly less than the
    # specified limit.
    output_byte_limit: Annotated[
        Optional[int],
        Field(
            alias="outputByteLimit",
            description="Maximum number of output bytes to retain.\n\nWhen the limit is exceeded, the Client truncates from the beginning of the output\nto stay within the limit.\n\nThe Client MUST ensure truncation happens at a character boundary to maintain valid\nstring output, even if this means the retained output is slightly less than the\nspecified limit.",
            ge=0,
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # A human-readable message describing what input is needed.
    message: Annotated[
        str,
        Field(description="A human-readable message describing what input is needed."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    mode: Literal["url"]
    # The unique identifier for this elicitation.
    elicitation_id: Annotated[
        str,
        Field(
            alias="elicitationId",
            description="The unique identifier for this elicitation.",
        ),
    ]
    # The URL to direct the user to.
    url: Annotated[AnyUrl, Field(description="The URL to direct the user to.")]


class CreateUrlRequestElicitationRequest(ElicitationRequestScope):
    # A human-readable message describing what input is needed.
    message: Annotated[
        str,
        Field(description="A human-readable message describing what input is needed."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    mode: Literal["url"]
    # The unique identifier for this elicitation.
    elicitation_id: Annotated[
        str,
        Field(
            alias="elicitationId",
            description="The unique identifier for this elicitation.",
        ),
    ]
    # The URL to direct the user to.
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
    # Optional title for the property.
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None
    # Human-readable description.
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    # Minimum number of items to select.
    min_items: Annotated[
        Optional[int],
        Field(alias="minItems", description="Minimum number of items to select.", ge=0),
    ] = None
    # Maximum number of items to select.
    max_items: Annotated[
        Optional[int],
        Field(alias="maxItems", description="Maximum number of items to select.", ge=0),
    ] = None
    # The items definition describing allowed values.
    items: Annotated[
        Union[StringMultiSelectItems, OtherMultiSelectItems, TitledMultiSelectItems],
        Field(description="The items definition describing allowed values."),
    ]
    # Default selected values.
    default: Annotated[Optional[List[str]], Field(description="Default selected values.")] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The ACP MCP server ID that was provided by the component declaring the MCP server.
    server_id: Annotated[
        str,
        Field(
            alias="serverId",
            description="The ACP MCP server ID that was provided by the component declaring the MCP server.",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class MessageMcpRequest(BaseModel):
    # The MCP-over-ACP connection this message is sent on.
    connection_id: Annotated[
        str,
        Field(
            alias="connectionId",
            description="The MCP-over-ACP connection this message is sent on.",
        ),
    ]
    # The inner MCP method name.
    method: Annotated[str, Field(description="The inner MCP method name.")]
    # Optional inner MCP params.
    #
    # If omitted or set to `null`, the inner MCP message has no params.
    params: Annotated[
        Optional[Dict[str, Any]],
        Field(
            description="Optional inner MCP params.\n\nIf omitted or set to `null`, the inner MCP message has no params."
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionCapabilities(BaseModel):
    # Whether the agent supports `session/list`.
    #
    # Optional. Omitted or `null` both mean the agent does not advertise support.
    # Supplying `{}` means the agent supports listing sessions.
    list: Annotated[
        Optional[SessionListCapabilities],
        Field(
            description="Whether the agent supports `session/list`.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports listing sessions."
        ),
    ] = None
    # Whether the agent supports `session/delete`.
    #
    # Optional. Omitted or `null` both mean the agent does not advertise support.
    # Supplying `{}` means the agent supports deleting sessions from `session/list`.
    delete: Annotated[
        Optional[SessionDeleteCapabilities],
        Field(
            description="Whether the agent supports `session/delete`.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports deleting sessions from `session/list`."
        ),
    ] = None
    # Whether the agent supports `additionalDirectories` on supported session lifecycle requests.
    #
    # Optional. Omitted or `null` both mean the agent does not advertise support.
    # Supplying `{}` means the agent supports `additionalDirectories` on
    # supported session lifecycle requests.
    #
    # Agents that also support `session/list` may return
    # `SessionInfo.additionalDirectories` to report the complete ordered
    # additional-root list associated with a listed session.
    additional_directories: Annotated[
        Optional[SessionAdditionalDirectoriesCapabilities],
        Field(
            alias="additionalDirectories",
            description="Whether the agent supports `additionalDirectories` on supported session lifecycle requests.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports `additionalDirectories` on\nsupported session lifecycle requests.\n\nAgents that also support `session/list` may return\n`SessionInfo.additionalDirectories` to report the complete ordered\nadditional-root list associated with a listed session.",
        ),
    ] = None
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Whether the agent supports `session/fork`.
    #
    # Optional. Omitted or `null` both mean the agent does not advertise support.
    # Supplying `{}` means the agent supports forking sessions.
    fork: Annotated[
        Optional[SessionForkCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nWhether the agent supports `session/fork`.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports forking sessions."
        ),
    ] = None
    # Whether the agent supports `session/resume`.
    #
    # Optional. Omitted or `null` both mean the agent does not advertise support.
    # Supplying `{}` means the agent supports resuming sessions.
    resume: Annotated[
        Optional[SessionResumeCapabilities],
        Field(
            description="Whether the agent supports `session/resume`.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports resuming sessions."
        ),
    ] = None
    # Whether the agent supports `session/close`.
    #
    # Optional. Omitted or `null` both mean the agent does not advertise support.
    # Supplying `{}` means the agent supports closing sessions.
    close: Annotated[
        Optional[SessionCloseCapabilities],
        Field(
            description="Whether the agent supports `session/close`.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports closing sessions."
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Whether the agent supports the logout method.
    #
    # Optional. Omitted or `null` both mean the agent does not advertise support.
    # Supplying `{}` means the agent supports the logout method.
    logout: Annotated[
        Optional[LogoutCapabilities],
        Field(
            description="Whether the agent supports the logout method.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports the logout method."
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The sync kind the agent wants: `"full"` or `"incremental"`.
    sync_kind: Annotated[
        str,
        Field(
            alias="syncKind",
            description='The sync kind the agent wants: `"full"` or `"incremental"`.',
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesContextCapabilities(BaseModel):
    # Whether the agent wants recent files context.
    recent_files: Annotated[
        Optional[NesRecentFilesCapabilities],
        Field(
            alias="recentFiles",
            description="Whether the agent wants recent files context.",
        ),
    ] = None
    # Whether the agent wants related snippets context.
    related_snippets: Annotated[
        Optional[NesRelatedSnippetsCapabilities],
        Field(
            alias="relatedSnippets",
            description="Whether the agent wants related snippets context.",
        ),
    ] = None
    # Whether the agent wants edit history context.
    edit_history: Annotated[
        Optional[NesEditHistoryCapabilities],
        Field(
            alias="editHistory",
            description="Whether the agent wants edit history context.",
        ),
    ] = None
    # Whether the agent wants user actions context.
    user_actions: Annotated[
        Optional[NesUserActionsCapabilities],
        Field(
            alias="userActions",
            description="Whether the agent wants user actions context.",
        ),
    ] = None
    # Whether the agent wants open files context.
    open_files: Annotated[
        Optional[NesOpenFilesCapabilities],
        Field(alias="openFiles", description="Whether the agent wants open files context."),
    ] = None
    # Whether the agent wants diagnostics context.
    diagnostics: Annotated[
        Optional[NesDiagnosticsCapabilities],
        Field(description="Whether the agent wants diagnostics context."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Provider identifier, for example "main" or "openai".
    provider_id: Annotated[
        str,
        Field(
            alias="providerId",
            description='Provider identifier, for example "main" or "openai".',
        ),
    ]
    # Supported protocol types for this provider.
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
    # Whether this provider is mandatory and cannot be disabled via `providers/disable`.
    # If true, clients must not call `providers/disable` for this provider ID.
    required: Annotated[
        bool,
        Field(
            description="Whether this provider is mandatory and cannot be disabled via `providers/disable`.\nIf true, clients must not call `providers/disable` for this provider ID."
        ),
    ]
    # Current effective non-secret routing config.
    # Null or omitted means provider is disabled.
    current: Annotated[
        Optional[ProviderCurrentConfig],
        Field(description="Current effective non-secret routing config.\nNull or omitted means provider is disabled."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The current mode the Agent is in.
    current_mode_id: Annotated[
        str,
        Field(alias="currentModeId", description="The current mode the Agent is in."),
    ]
    # The set of modes that the Agent can operate in
    available_modes: Annotated[
        List[SessionMode],
        Field(
            alias="availableModes",
            description="The set of modes that the Agent can operate in",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Unique identifier for the configuration option.
    id: Annotated[str, Field(description="Unique identifier for the configuration option.")]
    # Human-readable label for the option.
    name: Annotated[str, Field(description="Human-readable label for the option.")]
    # Optional description for the Client to display to the user.
    description: Annotated[
        Optional[str],
        Field(description="Optional description for the Client to display to the user."),
    ] = None
    # Optional semantic category for this option (UX only).
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Unique identifier for this group.
    group: Annotated[str, Field(description="Unique identifier for this group.")]
    # Human-readable label for this group.
    name: Annotated[str, Field(description="Human-readable label for this group.")]
    # The set of option values in this group.
    options: Annotated[
        List[SessionConfigSelectOption],
        Field(description="The set of option values in this group."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Array of session information objects
    sessions: Annotated[List[SessionInfo], Field(description="Array of session information objects")]
    # Opaque cursor token. If present, pass this in the next request's cursor parameter
    # to fetch the next page. If absent, there are no more results.
    next_cursor: Annotated[
        Optional[str],
        Field(
            alias="nextCursor",
            description="Opaque cursor token. If present, pass this in the next request's cursor parameter\nto fetch the next page. If absent, there are no more results.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Indicates why the agent stopped processing the turn.
    stop_reason: Annotated[
        StopReason,
        Field(
            alias="stopReason",
            description="Indicates why the agent stopped processing the turn.",
        ),
    ]
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Token usage for this turn (optional).
    usage: Annotated[
        Optional[Usage],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nToken usage for this turn (optional)."
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The start position (inclusive).
    start: Annotated[Position, Field(description="The start position (inclusive).")]
    # The end position (exclusive).
    end: Annotated[Position, Field(description="The end position (exclusive).")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class Error(BaseModel):
    # A number indicating the error type that occurred.
    # This must be an integer as defined in the JSON-RPC specification.
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
    # A string providing a short description of the error.
    # The message should be limited to a concise single sentence.
    message: Annotated[
        str,
        Field(
            description="A string providing a short description of the error.\nThe message should be limited to a concise single sentence."
        ),
    ]
    # Optional primitive or structured value that contains additional information about the error.
    # This may include debugging information or context-specific details.
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
    # Human-readable description of what this task aims to accomplish.
    content: Annotated[
        str,
        Field(description="Human-readable description of what this task aims to accomplish."),
    ]
    # The relative importance of this task.
    # Used to indicate which tasks are most critical to the overall goal.
    priority: Annotated[
        PlanEntryPriority,
        Field(
            description="The relative importance of this task.\nUsed to indicate which tasks are most critical to the overall goal."
        ),
    ]
    # Current execution status of this task.
    status: Annotated[PlanEntryStatus, Field(description="Current execution status of this task.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class Plan(BaseModel):
    # The list of tasks to be accomplished.
    #
    # When updating a plan, the agent must send a complete list of all entries
    # with their current status. The client replaces the entire plan with each update.
    entries: Annotated[
        List[PlanEntry],
        Field(
            description="The list of tasks to be accomplished.\n\nWhen updating a plan, the agent must send a complete list of all entries\nwith their current status. The client replaces the entire plan with each update."
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The plan ID to update.
    plan_id: Annotated[str, Field(alias="planId", description="The plan ID to update.")]
    # The list of tasks to be accomplished.
    #
    # When updating an item-based plan, the agent must send a complete list of all entries
    # with their current status. The client replaces that plan with each update.
    entries: Annotated[
        List[PlanEntry],
        Field(
            description="The list of tasks to be accomplished.\n\nWhen updating an item-based plan, the agent must send a complete list of all entries\nwith their current status. The client replaces that plan with each update."
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The input specification for a command.
    root: Annotated[
        UnstructuredCommandInput,
        Field(description="The input specification for a command."),
    ]


class SessionConfigOptionsCapabilities(BaseModel):
    # Whether the client supports boolean session configuration options.
    #
    # Optional. Omitted or `null` both mean the client does not advertise support.
    # Supplying `{}` means agents may include `type: "boolean"` entries in
    # `configOptions`, and the client may send `session/set_config_option`
    # requests with `type: "boolean"` and a boolean `value`.
    boolean: Annotated[
        Optional[BooleanConfigOptionCapabilities],
        Field(
            description='Whether the client supports boolean session configuration options.\n\nOptional. Omitted or `null` both mean the client does not advertise support.\nSupplying `{}` means agents may include `type: "boolean"` entries in\n`configOptions`, and the client may send `session/set_config_option`\nrequests with `type: "boolean"` and a boolean `value`.'
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Whether the client supports form-based elicitation.
    #
    # Optional. Omitted or `null` both mean the client does not advertise support.
    # Supplying `{}` means the client supports form-based elicitation.
    form: Annotated[
        Optional[ElicitationFormCapabilities],
        Field(
            description="Whether the client supports form-based elicitation.\n\nOptional. Omitted or `null` both mean the client does not advertise support.\nSupplying `{}` means the client supports form-based elicitation."
        ),
    ] = None
    # Whether the client supports URL-based elicitation.
    #
    # Optional. Omitted or `null` both mean the client does not advertise support.
    # Supplying `{}` means the client supports URL-based elicitation.
    url: Annotated[
        Optional[ElicitationUrlCapabilities],
        Field(
            description="Whether the client supports URL-based elicitation.\n\nOptional. Omitted or `null` both mean the client does not advertise support.\nSupplying `{}` means the client supports URL-based elicitation."
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Whether the client supports the `jump` suggestion kind.
    jump: Annotated[
        Optional[NesJumpCapabilities],
        Field(description="Whether the client supports the `jump` suggestion kind."),
    ] = None
    # Whether the client supports the `rename` suggestion kind.
    rename: Annotated[
        Optional[NesRenameCapabilities],
        Field(description="Whether the client supports the `rename` suggestion kind."),
    ] = None
    # Whether the client supports the `searchAndReplace` suggestion kind.
    search_and_replace: Annotated[
        Optional[NesSearchAndReplaceCapabilities],
        Field(
            alias="searchAndReplace",
            description="Whether the client supports the `searchAndReplace` suggestion kind.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # List of MCP servers to connect to for this session.
    mcp_servers: Annotated[
        List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]],
        Field(
            alias="mcpServers",
            description="List of MCP servers to connect to for this session.",
        ),
    ]
    # The working directory for this session. Must be an absolute path.
    cwd: Annotated[
        str,
        Field(description="The working directory for this session. Must be an absolute path."),
    ]
    # Additional workspace roots to activate for this session. Each path must be absolute.
    #
    # When omitted or empty, no additional roots are activated. When non-empty,
    # this is the complete resulting additional-root list for the loaded
    # session. It may differ from any previously used or reported list as long as
    # the request `cwd` matches the session's `cwd`.
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="Additional workspace roots to activate for this session. Each path must be absolute.\n\nWhen omitted or empty, no additional roots are activated. When non-empty,\nthis is the complete resulting additional-root list for the loaded\nsession. It may differ from any previously used or reported list as long as\nthe request `cwd` matches the session's `cwd`.",
        ),
    ] = None
    # The ID of the session to load.
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to load.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The ID of the session to fork.
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to fork.")]
    # The working directory for this session. Must be an absolute path.
    cwd: Annotated[
        str,
        Field(description="The working directory for this session. Must be an absolute path."),
    ]
    # Additional workspace roots to activate for this session. Each path must be absolute.
    #
    # When omitted or empty, no additional roots are activated. When non-empty,
    # this is the complete resulting additional-root list for the forked
    # session.
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="Additional workspace roots to activate for this session. Each path must be absolute.\n\nWhen omitted or empty, no additional roots are activated. When non-empty,\nthis is the complete resulting additional-root list for the forked\nsession.",
        ),
    ] = None
    # List of MCP servers to connect to for this session.
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]]],
        Field(
            alias="mcpServers",
            description="List of MCP servers to connect to for this session.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The ID of the session to resume.
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to resume.")]
    # The working directory for this session. Must be an absolute path.
    cwd: Annotated[
        str,
        Field(description="The working directory for this session. Must be an absolute path."),
    ]
    # Additional workspace roots to activate for this session. Each path must be absolute.
    #
    # When omitted or empty, no additional roots are activated. When non-empty,
    # this is the complete resulting additional-root list for the resumed
    # session. It may differ from any previously used or reported list as long as
    # the request `cwd` matches the session's `cwd`.
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="Additional workspace roots to activate for this session. Each path must be absolute.\n\nWhen omitted or empty, no additional roots are activated. When non-empty,\nthis is the complete resulting additional-root list for the resumed\nsession. It may differ from any previously used or reported list as long as\nthe request `cwd` matches the session's `cwd`.",
        ),
    ] = None
    # List of MCP servers to connect to for this session.
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]]],
        Field(
            alias="mcpServers",
            description="List of MCP servers to connect to for this session.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The root URI of the workspace.
    workspace_uri: Annotated[
        Optional[str],
        Field(alias="workspaceUri", description="The root URI of the workspace."),
    ] = None
    # The workspace folders.
    workspace_folders: Annotated[
        Optional[List[WorkspaceFolder]],
        Field(alias="workspaceFolders", description="The workspace folders."),
    ] = None
    # Repository metadata, if the workspace is a git repository.
    repository: Annotated[
        Optional[NesRepository],
        Field(description="Repository metadata, if the workspace is a git repository."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The URI of the file containing the snippets.
    uri: Annotated[str, Field(description="The URI of the file containing the snippets.")]
    # The code excerpts.
    excerpts: Annotated[List[NesExcerpt], Field(description="The code excerpts.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesOpenFile(BaseModel):
    # The URI of the file.
    uri: Annotated[str, Field(description="The URI of the file.")]
    # The language identifier.
    language_id: Annotated[str, Field(alias="languageId", description="The language identifier.")]
    # The visible range in the editor, if any.
    visible_range: Annotated[
        Optional[Range],
        Field(alias="visibleRange", description="The visible range in the editor, if any."),
    ] = None
    # Timestamp in milliseconds since epoch of when the file was last focused.
    last_focused_ms: Annotated[
        Optional[int],
        Field(
            alias="lastFocusedMs",
            description="Timestamp in milliseconds since epoch of when the file was last focused.",
            ge=0,
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The URI of the file containing the diagnostic.
    uri: Annotated[str, Field(description="The URI of the file containing the diagnostic.")]
    # The range of the diagnostic.
    range: Annotated[Range, Field(description="The range of the diagnostic.")]
    # The severity of the diagnostic.
    severity: Annotated[str, Field(description="The severity of the diagnostic.")]
    # The diagnostic message.
    message: Annotated[str, Field(description="The diagnostic message.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ClientErrorMessage(BaseModel):
    # The id of the request this response answers.
    id: Annotated[
        Optional[Union[int, str]],
        Field(description="The id of the request this response answers."),
    ]
    # Method-specific error data.
    error: Annotated[Error, Field(description="Method-specific error data.")]


class AllowedOutcome(SelectedPermissionOutcome):
    outcome: Literal["selected"]


class TerminalOutputResponse(BaseModel):
    # The terminal output captured so far.
    output: Annotated[str, Field(description="The terminal output captured so far.")]
    # Whether the output was truncated due to byte limits.
    truncated: Annotated[bool, Field(description="Whether the output was truncated due to byte limits.")]
    # Exit status if the command has completed.
    exit_status: Annotated[
        Optional[TerminalExitStatus],
        Field(alias="exitStatus", description="Exit status if the command has completed."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    action: Literal["accept"]


class TextDocumentContentChangeEvent(BaseModel):
    # The range of the document that changed. If `None`, the entire content is replaced.
    range: Annotated[
        Optional[Range],
        Field(description="The range of the document that changed. If `None`, the entire content is replaced."),
    ] = None
    # The new text for the range, or the full document content if `range` is `None`.
    text: Annotated[
        str,
        Field(description="The new text for the range, or the full document content if `range` is `None`."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DidFocusDocumentNotification(BaseModel):
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    # The URI of the focused document.
    uri: Annotated[str, Field(description="The URI of the focused document.")]
    # The version number of the document.
    version: Annotated[int, Field(description="The version number of the document.")]
    # The current cursor position.
    position: Annotated[Position, Field(description="The current cursor position.")]
    # The portion of the file currently visible in the editor viewport.
    visible_range: Annotated[
        Range,
        Field(
            alias="visibleRange",
            description="The portion of the file currently visible in the editor viewport.",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class RejectNesNotification(BaseModel):
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    # The ID of the rejected suggestion.
    id: Annotated[str, Field(description="The ID of the rejected suggestion.")]
    # The reason for rejection.
    reason: Annotated[Optional[str], Field(description="The reason for rejection.")] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The actual content block.
    content: Annotated[
        Union[
            TextContentBlock, ImageContentBlock, AudioContentBlock, ResourceContentBlock, EmbeddedResourceContentBlock
        ],
        Field(description="The actual content block.", discriminator="type"),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The id of the request this response answers.
    id: Annotated[
        Optional[Union[int, str]],
        Field(description="The id of the request this response answers."),
    ]
    # Method-specific error data.
    error: Annotated[Error, Field(description="Method-specific error data.")]


class NesDocumentEventCapabilities(BaseModel):
    # Whether the agent wants `document/didOpen` events.
    did_open: Annotated[
        Optional[NesDocumentDidOpenCapabilities],
        Field(
            alias="didOpen",
            description="Whether the agent wants `document/didOpen` events.",
        ),
    ] = None
    # Whether the agent wants `document/didChange` events, and the sync kind.
    did_change: Annotated[
        Optional[NesDocumentDidChangeCapabilities],
        Field(
            alias="didChange",
            description="Whether the agent wants `document/didChange` events, and the sync kind.",
        ),
    ] = None
    # Whether the agent wants `document/didClose` events.
    did_close: Annotated[
        Optional[NesDocumentDidCloseCapabilities],
        Field(
            alias="didClose",
            description="Whether the agent wants `document/didClose` events.",
        ),
    ] = None
    # Whether the agent wants `document/didSave` events.
    did_save: Annotated[
        Optional[NesDocumentDidSaveCapabilities],
        Field(
            alias="didSave",
            description="Whether the agent wants `document/didSave` events.",
        ),
    ] = None
    # Whether the agent wants `document/didFocus` events.
    did_focus: Annotated[
        Optional[NesDocumentDidFocusCapabilities],
        Field(
            alias="didFocus",
            description="Whether the agent wants `document/didFocus` events.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Configurable providers with current routing info suitable for UI display.
    providers: Annotated[
        List[ProviderInfo],
        Field(description="Configurable providers with current routing info suitable for UI display."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class SessionConfigSelect(BaseModel):
    # The currently selected value.
    current_value: Annotated[str, Field(alias="currentValue", description="The currently selected value.")]
    # The set of selectable options.
    options: Annotated[
        Union[List[SessionConfigSelectOption], List[SessionConfigSelectGroup]],
        Field(description="The set of selectable options."),
    ]


class NesTextEdit(BaseModel):
    # The range to replace.
    range: Annotated[Range, Field(description="The range to replace.")]
    # The replacement text.
    new_text: Annotated[str, Field(alias="newText", description="The replacement text.")]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesEditSuggestion(BaseModel):
    # Unique identifier for accept/reject tracking.
    id: Annotated[str, Field(description="Unique identifier for accept/reject tracking.")]
    # The URI of the file to edit.
    uri: Annotated[str, Field(description="The URI of the file to edit.")]
    # The text edits to apply.
    edits: Annotated[List[NesTextEdit], Field(description="The text edits to apply.")]
    # Optional suggested cursor position after applying edits.
    cursor_position: Annotated[
        Optional[Position],
        Field(
            alias="cursorPosition",
            description="Optional suggested cursor position after applying edits.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # A single item of content
    content: Annotated[
        Union[
            TextContentBlock, ImageContentBlock, AudioContentBlock, ResourceContentBlock, EmbeddedResourceContentBlock
        ],
        Field(description="A single item of content", discriminator="type"),
    ]
    # A unique identifier for the message this chunk belongs to.
    #
    # All chunks belonging to the same message share the same `messageId`.
    # A change in `messageId` indicates a new message has started.
    message_id: Annotated[
        Optional[str],
        Field(
            alias="messageId",
            description="A unique identifier for the message this chunk belongs to.\n\nAll chunks belonging to the same message share the same `messageId`.\nA change in `messageId` indicates a new message has started.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The updated plan content.
    plan: Annotated[
        Union[PlanUpdateItems, PlanUpdateFile, PlanUpdateMarkdown],
        Field(description="The updated plan content.", discriminator="type"),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class AvailableCommand(BaseModel):
    # Command name (e.g., `create_plan`, `research_codebase`).
    name: Annotated[
        str,
        Field(description="Command name (e.g., `create_plan`, `research_codebase`)."),
    ]
    # Human-readable description of what the command does.
    description: Annotated[str, Field(description="Human-readable description of what the command does.")]
    # Input for the command if required
    input: Annotated[
        Optional[AvailableCommandInput],
        Field(description="Input for the command if required"),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Commands the agent can execute
    available_commands: Annotated[
        List[AvailableCommand],
        Field(alias="availableCommands", description="Commands the agent can execute"),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Config option capabilities supported by the client.
    #
    # Omitted or `null` both mean the client does not advertise support for any
    # config option extensions.
    config_options: Annotated[
        Optional[SessionConfigOptionsCapabilities],
        Field(
            alias="configOptions",
            description="Config option capabilities supported by the client.\n\nOmitted or `null` both mean the client does not advertise support for any\nconfig option extensions.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The working directory for this session. Must be an absolute path.
    cwd: Annotated[
        str,
        Field(description="The working directory for this session. Must be an absolute path."),
    ]
    # Additional workspace roots for this session. Each path must be absolute.
    #
    # These expand the session's filesystem scope without changing `cwd`, which
    # remains the base for relative paths. When omitted or empty, no
    # additional roots are activated for the new session.
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="Additional workspace roots for this session. Each path must be absolute.\n\nThese expand the session's filesystem scope without changing `cwd`, which\nremains the base for relative paths. When omitted or empty, no\nadditional roots are activated for the new session.",
        ),
    ] = None
    # List of MCP (Model Context Protocol) servers the agent should connect to.
    mcp_servers: Annotated[
        List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]],
        Field(
            alias="mcpServers",
            description="List of MCP (Model Context Protocol) servers the agent should connect to.",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The ID of the session to send this user message to
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session to send this user message to",
        ),
    ]
    # The blocks of content that compose the user's message.
    #
    # As a baseline, the Agent MUST support [`ContentBlock::Text`] and [`ContentBlock::ResourceLink`],
    # while other variants are optionally enabled via [`PromptCapabilities`].
    #
    # The Client MUST adapt its interface according to [`PromptCapabilities`].
    #
    # The client MAY include referenced pieces of context as either
    # [`ContentBlock::Resource`] or [`ContentBlock::ResourceLink`].
    #
    # When available, [`ContentBlock::Resource`] is preferred
    # as it avoids extra round-trips and allows the message to include
    # pieces of context from sources the agent may not have access to.
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class NesSuggestContext(BaseModel):
    # Recently accessed files.
    recent_files: Annotated[
        Optional[List[NesRecentFile]],
        Field(alias="recentFiles", description="Recently accessed files."),
    ] = None
    # Related code snippets.
    related_snippets: Annotated[
        Optional[List[NesRelatedSnippet]],
        Field(alias="relatedSnippets", description="Related code snippets."),
    ] = None
    # Recent edit history.
    edit_history: Annotated[
        Optional[List[NesEditHistoryEntry]],
        Field(alias="editHistory", description="Recent edit history."),
    ] = None
    # Recent user actions (typing, navigation, etc.).
    user_actions: Annotated[
        Optional[List[NesUserAction]],
        Field(
            alias="userActions",
            description="Recent user actions (typing, navigation, etc.).",
        ),
    ] = None
    # Currently open files in the editor.
    open_files: Annotated[
        Optional[List[NesOpenFile]],
        Field(alias="openFiles", description="Currently open files in the editor."),
    ] = None
    # Current diagnostics (errors, warnings).
    diagnostics: Annotated[
        Optional[List[NesDiagnostic]],
        Field(description="Current diagnostics (errors, warnings)."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class RequestPermissionResponse(BaseModel):
    # The user's decision on the permission request.
    outcome: Annotated[
        Union[DeniedOutcome, AllowedOutcome],
        Field(
            description="The user's decision on the permission request.",
            discriminator="outcome",
        ),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class DidChangeDocumentNotification(BaseModel):
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    # The URI of the changed document.
    uri: Annotated[str, Field(description="The URI of the changed document.")]
    # The new version number of the document.
    version: Annotated[int, Field(description="The new version number of the document.")]
    # The content changes.
    content_changes: Annotated[
        List[TextDocumentContentChangeEvent],
        Field(alias="contentChanges", description="The content changes."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Type discriminator. Always `"object"`.
    type: Annotated[Optional[str], Field(description='Type discriminator. Always `"object"`.')] = "object"
    # Optional title for the schema.
    title: Annotated[Optional[str], Field(description="Optional title for the schema.")] = None
    # Property definitions (must be primitive types).
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
    # List of required property names.
    required: Annotated[Optional[List[str]], Field(description="List of required property names.")] = None
    # Optional description of what this schema represents.
    description: Annotated[
        Optional[str],
        Field(description="Optional description of what this schema represents."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # A JSON Schema describing the form fields to present to the user.
    requested_schema: Annotated[
        ElicitationSchema,
        Field(
            alias="requestedSchema",
            description="A JSON Schema describing the form fields to present to the user.",
        ),
    ]


class ElicitationFormRequestMode(ElicitationRequestScope):
    # A JSON Schema describing the form fields to present to the user.
    requested_schema: Annotated[
        ElicitationSchema,
        Field(
            alias="requestedSchema",
            description="A JSON Schema describing the form fields to present to the user.",
        ),
    ]


class ElicitationFormMode(RootModel[Union[ElicitationFormSessionMode, ElicitationFormRequestMode]]):
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Form-based elicitation mode where the client renders a form from the provided schema.
    root: Annotated[
        Union[ElicitationFormSessionMode, ElicitationFormRequestMode],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nForm-based elicitation mode where the client renders a form from the provided schema."
        ),
    ]


class NesEventCapabilities(BaseModel):
    # Document event capabilities.
    document: Annotated[
        Optional[NesDocumentEventCapabilities],
        Field(description="Document event capabilities."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Unique identifier for the configuration option.
    id: Annotated[str, Field(description="Unique identifier for the configuration option.")]
    # Human-readable label for the option.
    name: Annotated[str, Field(description="Human-readable label for the option.")]
    # Optional description for the Client to display to the user.
    description: Annotated[
        Optional[str],
        Field(description="Optional description for the Client to display to the user."),
    ] = None
    # Optional semantic category for this option (UX only).
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Initial mode state if supported by the Agent
    #
    # See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    modes: Annotated[
        Optional[SessionModeState],
        Field(
            description="Initial mode state if supported by the Agent\n\nSee protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)"
        ),
    ] = None
    # Initial session configuration options if supported by the Agent.
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Unique identifier for the newly created forked session.
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="Unique identifier for the newly created forked session.",
        ),
    ]
    # Initial mode state if supported by the Agent
    #
    # See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    modes: Annotated[
        Optional[SessionModeState],
        Field(
            description="Initial mode state if supported by the Agent\n\nSee protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)"
        ),
    ] = None
    # Initial session configuration options if supported by the Agent.
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Initial mode state if supported by the Agent
    #
    # See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    modes: Annotated[
        Optional[SessionModeState],
        Field(
            description="Initial mode state if supported by the Agent\n\nSee protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)"
        ),
    ] = None
    # Initial session configuration options if supported by the Agent.
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The full set of configuration options and their current values.
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Unique identifier for this tool call within the session.
    tool_call_id: Annotated[
        str,
        Field(
            alias="toolCallId",
            description="Unique identifier for this tool call within the session.",
        ),
    ]
    # Human-readable title describing what the tool is doing.
    title: Annotated[
        str,
        Field(description="Human-readable title describing what the tool is doing."),
    ]
    # The category of tool being invoked.
    # Helps clients choose appropriate icons and UI treatment.
    kind: Annotated[
        Optional[ToolKind],
        Field(
            description="The category of tool being invoked.\nHelps clients choose appropriate icons and UI treatment."
        ),
    ] = None
    # Current execution status of the tool call.
    status: Annotated[Optional[ToolCallStatus], Field(description="Current execution status of the tool call.")] = None
    # Content produced by the tool call.
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
    # File locations affected by this tool call.
    # Enables "follow-along" features in clients.
    locations: Annotated[
        Optional[List[ToolCallLocation]],
        Field(description='File locations affected by this tool call.\nEnables "follow-along" features in clients.'),
    ] = None
    # Raw input parameters sent to the tool.
    raw_input: Annotated[
        Optional[Any],
        Field(alias="rawInput", description="Raw input parameters sent to the tool."),
    ] = None
    # Raw output returned by the tool.
    raw_output: Annotated[
        Optional[Any],
        Field(alias="rawOutput", description="Raw output returned by the tool."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The full set of configuration options and their current values.
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # File system capabilities supported by the client.
    # Determines which file operations the agent can request.
    fs: Annotated[
        Optional[FileSystemCapabilities],
        Field(
            description="File system capabilities supported by the client.\nDetermines which file operations the agent can request.",
            validate_default=True,
        ),
    ] = FileSystemCapabilities()
    # Whether the Client support all `terminal/*` methods.
    terminal: Annotated[
        Optional[bool],
        Field(description="Whether the Client support all `terminal/*` methods."),
    ] = False
    # Session-related capabilities supported by the client.
    #
    # Optional. Omitted or `null` both mean the client does not advertise any
    # session-related extensions.
    session: Annotated[
        Optional[ClientSessionCapabilities],
        Field(
            description="Session-related capabilities supported by the client.\n\nOptional. Omitted or `null` both mean the client does not advertise any\nsession-related extensions."
        ),
    ] = None
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Whether the client supports `plan_update` and `plan_removed` session updates.
    #
    # Optional. Omitted or `null` both mean the client does not advertise support.
    # Supplying `{}` means the client can receive both update types.
    plan: Annotated[
        Optional[PlanCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nWhether the client supports `plan_update` and `plan_removed` session updates.\n\nOptional. Omitted or `null` both mean the client does not advertise support.\nSupplying `{}` means the client can receive both update types."
        ),
    ] = None
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Authentication capabilities supported by the client.
    # Determines which authentication method types the agent may include
    # in its `InitializeResponse`.
    auth: Annotated[
        Optional[AuthCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nAuthentication capabilities supported by the client.\nDetermines which authentication method types the agent may include\nin its `InitializeResponse`.",
            validate_default=True,
        ),
    ] = {"terminal": False}
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Elicitation capabilities supported by the client.
    # Determines which elicitation modes the agent may use.
    #
    # Optional. Omitted or `null` both mean the client does not advertise
    # elicitation support.
    elicitation: Annotated[
        Optional[ElicitationCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nElicitation capabilities supported by the client.\nDetermines which elicitation modes the agent may use.\n\nOptional. Omitted or `null` both mean the client does not advertise\nelicitation support."
        ),
    ] = None
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # NES (Next Edit Suggestions) capabilities supported by the client.
    #
    # Optional. Omitted or `null` both mean the client does not advertise any
    # NES suggestion-kind extensions.
    nes: Annotated[
        Optional[ClientNesCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nNES (Next Edit Suggestions) capabilities supported by the client.\n\nOptional. Omitted or `null` both mean the client does not advertise any\nNES suggestion-kind extensions."
        ),
    ] = None
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # The position encodings supported by the client, in order of preference.
    position_encodings: Annotated[
        Optional[List[str]],
        Field(
            alias="positionEncodings",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nThe position encodings supported by the client, in order of preference.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # The URI of the document to suggest for.
    uri: Annotated[str, Field(description="The URI of the document to suggest for.")]
    # The version number of the document.
    version: Annotated[int, Field(description="The version number of the document.")]
    # The current cursor position.
    position: Annotated[Position, Field(description="The current cursor position.")]
    # The current text selection range, if any.
    selection: Annotated[Optional[Range], Field(description="The current text selection range, if any.")] = None
    # What triggered this suggestion request.
    trigger_kind: Annotated[
        str,
        Field(alias="triggerKind", description="What triggered this suggestion request."),
    ]
    # Context for the suggestion, included based on agent capabilities.
    context: Annotated[
        Optional[NesSuggestContext],
        Field(description="Context for the suggestion, included based on agent capabilities."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ClientResponseMessage(BaseModel):
    # The id of the request this response answers.
    id: Annotated[
        Optional[Union[int, str]],
        Field(description="The id of the request this response answers."),
    ]
    # Method-specific response data.
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
    # A JSON-RPC response object.
    root: Annotated[
        Union[ClientResponseMessage, ClientErrorMessage],
        Field(description="A JSON-RPC response object."),
    ]


class ClientNotification(BaseModel):
    # The notification method name.
    method: Annotated[str, Field(description="The notification method name.")]
    # Method-specific notification parameters.
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
    # The ID of the tool call being updated.
    tool_call_id: Annotated[
        str,
        Field(alias="toolCallId", description="The ID of the tool call being updated."),
    ]
    # Update the tool kind.
    kind: Annotated[Optional[ToolKind], Field(description="Update the tool kind.")] = None
    # Update the execution status.
    status: Annotated[Optional[ToolCallStatus], Field(description="Update the execution status.")] = None
    # Update the human-readable title.
    title: Annotated[Optional[str], Field(description="Update the human-readable title.")] = None
    # Replace the content collection.
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
    # Replace the locations collection.
    locations: Annotated[
        Optional[List[ToolCallLocation]],
        Field(description="Replace the locations collection."),
    ] = None
    # Update the raw input.
    raw_input: Annotated[Optional[Any], Field(alias="rawInput", description="Update the raw input.")] = None
    # Update the raw output.
    raw_output: Annotated[Optional[Any], Field(alias="rawOutput", description="Update the raw output.")] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # A human-readable message describing what input is needed.
    message: Annotated[
        str,
        Field(description="A human-readable message describing what input is needed."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    mode: Literal["form"]
    # A JSON Schema describing the form fields to present to the user.
    requested_schema: Annotated[
        ElicitationSchema,
        Field(
            alias="requestedSchema",
            description="A JSON Schema describing the form fields to present to the user.",
        ),
    ]


class CreateFormRequestElicitationRequest(ElicitationRequestScope):
    # A human-readable message describing what input is needed.
    message: Annotated[
        str,
        Field(description="A human-readable message describing what input is needed."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None
    mode: Literal["form"]
    # A JSON Schema describing the form fields to present to the user.
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
    # Events the agent wants to receive.
    events: Annotated[
        Optional[NesEventCapabilities],
        Field(description="Events the agent wants to receive."),
    ] = None
    # Context the agent wants attached to each suggestion request.
    context: Annotated[
        Optional[NesContextCapabilities],
        Field(description="Context the agent wants attached to each suggestion request."),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # Unique identifier for the created session.
    #
    # Used in all subsequent requests for this conversation.
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="Unique identifier for the created session.\n\nUsed in all subsequent requests for this conversation.",
        ),
    ]
    # Initial mode state if supported by the Agent
    #
    # See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    modes: Annotated[
        Optional[SessionModeState],
        Field(
            description="Initial mode state if supported by the Agent\n\nSee protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)"
        ),
    ] = None
    # Initial session configuration options if supported by the Agent.
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The list of suggestions.
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The latest protocol version supported by the client.
    protocol_version: Annotated[
        int,
        Field(
            alias="protocolVersion",
            description="The latest protocol version supported by the client.",
            ge=0,
            le=65535,
        ),
    ]
    # Capabilities supported by the client.
    client_capabilities: Annotated[
        Optional[ClientCapabilities],
        Field(
            alias="clientCapabilities",
            description="Capabilities supported by the client.",
            validate_default=True,
        ),
    ] = ClientCapabilities()
    # Information about the Client name and version sent to the Agent.
    #
    # Note: in future versions of the protocol, this will be required.
    client_info: Annotated[
        Optional[Implementation],
        Field(
            alias="clientInfo",
            description="Information about the Client name and version sent to the Agent.\n\nNote: in future versions of the protocol, this will be required.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # Details about the tool call requiring permission.
    tool_call: Annotated[
        ToolCallUpdate,
        Field(
            alias="toolCall",
            description="Details about the tool call requiring permission.",
        ),
    ]
    # Available permission options for the user to choose from.
    options: Annotated[
        List[PermissionOption],
        Field(description="Available permission options for the user to choose from."),
    ]
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class AgentCapabilities(BaseModel):
    # Whether the agent supports `session/load`.
    load_session: Annotated[
        Optional[bool],
        Field(
            alias="loadSession",
            description="Whether the agent supports `session/load`.",
        ),
    ] = False
    # Prompt capabilities supported by the agent.
    prompt_capabilities: Annotated[
        Optional[PromptCapabilities],
        Field(
            alias="promptCapabilities",
            description="Prompt capabilities supported by the agent.",
            validate_default=True,
        ),
    ] = PromptCapabilities()
    # MCP capabilities supported by the agent.
    mcp_capabilities: Annotated[
        Optional[McpCapabilities],
        Field(
            alias="mcpCapabilities",
            description="MCP capabilities supported by the agent.",
            validate_default=True,
        ),
    ] = McpCapabilities()
    # Session lifecycle and prompt capabilities advertised by the agent.
    session_capabilities: Annotated[
        Optional[SessionCapabilities],
        Field(
            alias="sessionCapabilities",
            description="Session lifecycle and prompt capabilities advertised by the agent.",
            validate_default=True,
        ),
    ] = SessionCapabilities()
    # Authentication-related capabilities supported by the agent.
    auth: Annotated[
        Optional[AgentAuthCapabilities],
        Field(
            description="Authentication-related capabilities supported by the agent.",
            validate_default=True,
        ),
    ] = {}
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Provider configuration capabilities supported by the agent.
    #
    # Optional. Omitted or `null` both mean the agent does not advertise support.
    # Supplying `{}` means the agent supports provider configuration methods.
    providers: Annotated[
        Optional[ProvidersCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nProvider configuration capabilities supported by the agent.\n\nOptional. Omitted or `null` both mean the agent does not advertise support.\nSupplying `{}` means the agent supports provider configuration methods."
        ),
    ] = None
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # NES (Next Edit Suggestions) capabilities supported by the agent.
    #
    # Optional. Omitted or `null` both mean the agent does not advertise support
    # for NES methods.
    nes: Annotated[
        Optional[NesCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nNES (Next Edit Suggestions) capabilities supported by the agent.\n\nOptional. Omitted or `null` both mean the agent does not advertise support\nfor NES methods."
        ),
    ] = None
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # The position encoding selected by the agent from the client's supported encodings.
    position_encoding: Annotated[
        Optional[str],
        Field(
            alias="positionEncoding",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nThe position encoding selected by the agent from the client's supported encodings.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The ID of the session this update pertains to.
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session this update pertains to.",
        ),
    ]
    # The actual update content.
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
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
    field_meta: Annotated[
        Optional[Dict[str, Any]],
        Field(
            alias="_meta",
            description="The _meta property is reserved by ACP to allow clients and agents to attach additional\nmetadata to their interactions. Implementations MUST NOT make assumptions about values at\nthese keys.\n\nSee protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)",
        ),
    ] = None


class ClientRequest(BaseModel):
    # The request id used to correlate the matching response.
    id: Annotated[
        Optional[Union[int, str]],
        Field(description="The request id used to correlate the matching response."),
    ]
    # The method name to invoke.
    method: Annotated[str, Field(description="The method name to invoke.")]
    # Method-specific request parameters.
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
    # The request id used to correlate the matching response.
    id: Annotated[
        Optional[Union[int, str]],
        Field(description="The request id used to correlate the matching response."),
    ]
    # The method name to invoke.
    method: Annotated[str, Field(description="The method name to invoke.")]
    # Method-specific request parameters.
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
    # The protocol version the client specified if supported by the agent,
    # or the latest protocol version supported by the agent.
    #
    # The client should disconnect, if it doesn't support this version.
    protocol_version: Annotated[
        int,
        Field(
            alias="protocolVersion",
            description="The protocol version the client specified if supported by the agent,\nor the latest protocol version supported by the agent.\n\nThe client should disconnect, if it doesn't support this version.",
            ge=0,
            le=65535,
        ),
    ]
    # Capabilities supported by the agent.
    agent_capabilities: Annotated[
        Optional[AgentCapabilities],
        Field(
            alias="agentCapabilities",
            description="Capabilities supported by the agent.",
            validate_default=True,
        ),
    ] = AgentCapabilities()
    # Authentication methods supported by the agent.
    auth_methods: Annotated[
        Optional[List[Union[EnvVarAuthMethod, TerminalAuthMethod, AuthMethodAgent]]],
        Field(
            alias="authMethods",
            description="Authentication methods supported by the agent.",
            validate_default=True,
        ),
    ] = []
    # Information about the Agent name and version sent to the Client.
    #
    # Note: in future versions of the protocol, this will be required.
    agent_info: Annotated[
        Optional[Implementation],
        Field(
            alias="agentInfo",
            description="Information about the Agent name and version sent to the Client.\n\nNote: in future versions of the protocol, this will be required.",
        ),
    ] = None
    # The _meta property is reserved by ACP to allow clients and agents to attach additional
    # metadata to their interactions. Implementations MUST NOT make assumptions about values at
    # these keys.
    #
    # See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/extensibility)
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
    # The notification method name.
    method: Annotated[str, Field(description="The notification method name.")]
    # Method-specific notification parameters.
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
    # The id of the request this response answers.
    id: Annotated[
        Optional[Union[int, str]],
        Field(description="The id of the request this response answers."),
    ]
    # Method-specific response data.
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
    # A JSON-RPC response object.
    root: Annotated[
        Union[AgentResponseMessage, AgentErrorMessage],
        Field(description="A JSON-RPC response object."),
    ]
