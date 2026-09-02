# Generated from schema/v2/schema.json. Do not edit by hand.
# Schema ref: refs/tags/schema-v2.0.0-alpha.3

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from acp._deserialize import coerce_protocol_version, skip_invalid_items, use_default_on_error
from acp.experimental.v2._schema_base import BaseModel
from pydantic import (
    AnyUrl,
    AwareDatetime,
    ConfigDict,
    Field,
    RootModel,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
)


class OtherPermissionSubject(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: str
    """
    Custom or future permission subject type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class OtherToolCallContent(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: str
    """
    Custom or future tool call content type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class OtherContentBlock(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: str
    """
    Custom or future content block type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class TextResourceContents(BaseModel):
    text: str
    """
    Text payload carried by this content block.
    """
    uri: AnyUrl
    """
    URI associated with this resource or media payload.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type describing the encoded media payload.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("mime_type", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class BlobResourceContents(BaseModel):
    blob: Annotated[str, Field(json_schema_extra={"contentEncoding": "base64"})]
    """
    Base64-encoded bytes for a binary resource payload.
    """
    uri: AnyUrl
    """
    URI associated with this resource or media payload.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type describing the encoded media payload.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("mime_type", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class DiffPathChange(BaseModel):
    path: str
    """
    Absolute path for the operation.
    """


class DiffPathPairChange(BaseModel):
    old_path: Annotated[str, Field(alias="oldPath")]
    """
    Absolute path before the operation.
    """
    path: str
    """
    Absolute path after the operation.
    """


class Terminal(BaseModel):
    terminal_id: Annotated[str, Field(alias="terminalId")]
    """
    The ID of the terminal to display.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys. This metadata is scoped to the content reference. Omitted
    and `null` are equivalent and mean no item metadata was provided.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("line", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class CommandPermissionSubject(BaseModel):
    command: str
    """
    The command that would be run if permission is granted.
    """
    cwd: str
    """
    The absolute working directory for the command.
    """
    tool_call_id: Annotated[Optional[str], Field(alias="toolCallId")] = None
    """
    The associated tool call, when known. Omitted and `null` are equivalent.
    """
    terminal_id: Annotated[Optional[str], Field(alias="terminalId")] = None
    """
    The associated terminal, when already known. Omitted and `null` are equivalent.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys. Omitted and `null` are equivalent and mean no subject metadata was provided.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("terminal_id", "tool_call_id", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class CreateFormElicitationRequestBase(BaseModel):
    message: str
    """
    A human-readable message describing what input is needed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    mode: Literal["form"] = "form"


class CreateUrlElicitationRequestBase(BaseModel):
    message: str
    """
    A human-readable message describing what input is needed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    mode: Literal["url"] = "url"


class ElicitationSessionScope(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session this elicitation is tied to.
    """
    tool_call_id: Annotated[Optional[str], Field(alias="toolCallId")] = None
    """
    Optional tool call within the session.

    Optional. Omitted and `null` are equivalent and mean the elicitation is scoped to the
    session without a specific tool call.
    """

    @field_validator("tool_call_id", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    Optional. Omitted and `null` are equivalent and mean no description is provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class StringPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional title for the property.

    Optional. Omitted and `null` are equivalent and mean no title is provided.
    """
    description: Optional[str] = None
    """
    Human-readable description.

    Optional. Omitted and `null` are equivalent and mean no description is provided.
    """
    min_length: Annotated[Optional[int], Field(alias="minLength", ge=0)] = None
    """
    Minimum string length.

    Optional. Omitted and `null` are equivalent and mean there is no minimum length constraint.
    """
    max_length: Annotated[Optional[int], Field(alias="maxLength", ge=0)] = None
    """
    Maximum string length.

    Optional. Omitted and `null` are equivalent and mean there is no maximum length constraint.
    """
    pattern: Optional[str] = None
    """
    Pattern the string must match.

    Optional. Omitted and `null` are equivalent and mean there is no pattern constraint.
    """
    format: Optional[Union[Literal["email"], Literal["uri"], Literal["date"], Literal["date-time"], str]] = None
    """
    String format.

    Optional. Omitted and `null` are equivalent and mean there is no format constraint.
    """
    default: Optional[str] = None
    """
    Default value.

    Optional. Omitted and `null` are equivalent and mean no default value is provided.
    """
    enum: Annotated[Optional[List[str]], Field(min_length=1)] = None
    """
    Enum values for untitled single-select enums.
    Must contain at least one value when present.
    Optional. Omitted and `null` are equivalent and mean no untitled single-select choices are
    declared by `enum`.
    """
    one_of: Annotated[Optional[List[EnumOption]], Field(alias="oneOf", min_length=1)] = None
    """
    Titled enum options for titled single-select enums.
    Must contain at least one option when present.
    Optional. Omitted and `null` are equivalent and mean no titled single-select choices are
    declared by `oneOf`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class NumberPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional title for the property.

    Optional. Omitted and `null` are equivalent and mean no title is provided.
    """
    description: Optional[str] = None
    """
    Human-readable description.

    Optional. Omitted and `null` are equivalent and mean no description is provided.
    """
    minimum: Optional[float] = None
    """
    Minimum value (inclusive).

    Optional. Omitted and `null` are equivalent and mean there is no inclusive lower bound.
    """
    maximum: Optional[float] = None
    """
    Maximum value (inclusive).

    Optional. Omitted and `null` are equivalent and mean there is no inclusive upper bound.
    """
    default: Optional[float] = None
    """
    Default value.

    Optional. Omitted and `null` are equivalent and mean no default value is provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class IntegerPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional title for the property.

    Optional. Omitted and `null` are equivalent and mean no title is provided.
    """
    description: Optional[str] = None
    """
    Human-readable description.

    Optional. Omitted and `null` are equivalent and mean no description is provided.
    """
    minimum: Optional[int] = None
    """
    Minimum value (inclusive).

    Optional. Omitted and `null` are equivalent and mean there is no inclusive lower bound.
    """
    maximum: Optional[int] = None
    """
    Maximum value (inclusive).

    Optional. Omitted and `null` are equivalent and mean there is no inclusive upper bound.
    """
    default: Optional[int] = None
    """
    Default value.

    Optional. Omitted and `null` are equivalent and mean no default value is provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class BooleanPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional title for the property.

    Optional. Omitted and `null` are equivalent and mean no title is provided.
    """
    description: Optional[str] = None
    """
    Human-readable description.

    Optional. Omitted and `null` are equivalent and mean no description is provided.
    """
    default: Optional[bool] = None
    """
    Default value.

    Optional. Omitted and `null` are equivalent and mean no default value is provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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


class StringMultiSelectItemsBase(BaseModel):
    enum: Annotated[List[str], Field(min_length=1)]
    """
    Allowed enum values. Must contain at least one value.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class TitledMultiSelectItems(BaseModel):
    any_of: Annotated[List[EnumOption], Field(alias="anyOf", min_length=1)]
    """
    Titled enum options. Must contain at least one option.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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
    root: Union[ElicitationUrlSessionMode, ElicitationUrlRequestMode]
    """
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class PromptImageCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class PromptAudioCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class PromptEmbeddedContextCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class McpStdioCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class McpHttpCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class McpAcpCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionDeleteCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionAdditionalDirectoriesCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionForkCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class AgentAuthCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ProvidersCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesDocumentDidOpenCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesDocumentDidCloseCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesDocumentDidSaveCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesDocumentDidFocusCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("max_count", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class NesRelatedSnippetsCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("max_count", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("max_count", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class NesOpenFilesCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesDiagnosticsCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class AuthMethodTerminal(BaseModel):
    method_id: Annotated[str, Field(alias="methodId")]
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
    Additional arguments to append to the configured agent invocation for terminal auth.
    """
    env: Optional[List[EnvVariable]] = None
    """
    Additional environment variables to set on the configured agent invocation for terminal auth.
    Names MUST be unique. These values override same-named variables in the
    base launch configuration.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("args", "env", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class AuthMethodAgent(BaseModel):
    method_id: Annotated[str, Field(alias="methodId")]
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class LoginAuthResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ProviderCurrentConfig(BaseModel):
    api_type: Annotated[
        Union[Literal["anthropic"], Literal["openai"], Literal["azure"], Literal["vertex"], Literal["bedrock"], str],
        Field(alias="apiType"),
    ]
    """
    Protocol currently used by this provider.
    """
    base_url: Annotated[AnyUrl, Field(alias="baseUrl")]
    """
    Base URL currently used by this provider.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SetProviderResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class DisableProviderResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class LogoutAuthResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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
    updated_at: Annotated[Optional[AwareDatetime], Field(alias="updatedAt")] = None
    """
    RFC 3339 timestamp of last activity.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("title", "updated_at", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("additional_directories", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class DeleteSessionResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CloseSessionResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class PromptResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesJumpSuggestion(BaseModel):
    suggestion_id: Annotated[str, Field(alias="suggestionId")]
    """
    Unique identifier for accept/reject tracking.
    """
    uri: AnyUrl
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesRenameSuggestion(BaseModel):
    suggestion_id: Annotated[str, Field(alias="suggestionId")]
    """
    Unique identifier for accept/reject tracking.
    """
    uri: AnyUrl
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesSearchAndReplaceSuggestion(BaseModel):
    suggestion_id: Annotated[str, Field(alias="suggestionId")]
    """
    Unique identifier for accept/reject tracking.
    """
    uri: AnyUrl
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CloseNesResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class OtherSessionStateUpdateBase(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    state: str
    """
    Custom or future session state.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class SessionStateUpdateBase(BaseModel):
    session_update: Annotated[Literal["state_update"], Field(alias="sessionUpdate")] = "state_update"


class OtherSessionStateUpdate(OtherSessionStateUpdateBase, SessionStateUpdateBase):
    session_update: Annotated[Literal["state_update"], Field(alias="sessionUpdate")] = "state_update"


class OtherSessionUpdate(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    session_update: Annotated[str, Field(alias="sessionUpdate")]
    """
    Custom or future session update type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class RunningStateUpdate(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class Usage(BaseModel):
    total_tokens: Annotated[int, Field(alias="totalTokens", ge=0)]
    """
    Sum of all token types across session.
    """
    input_tokens: Annotated[int, Field(alias="inputTokens", ge=0)]
    """
    Total input tokens.
    """
    output_tokens: Annotated[int, Field(alias="outputTokens", ge=0)]
    """
    Total output tokens.
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("cached_read_tokens", "cached_write_tokens", "thought_tokens", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class IdleStateUpdate(BaseModel):
    stop_reason: Annotated[
        Optional[
            Union[
                Literal["end_turn"],
                Literal["max_tokens"],
                Literal["max_turn_requests"],
                Literal["refusal"],
                Literal["cancelled"],
                str,
            ]
        ],
        Field(alias="stopReason"),
    ] = None
    """
    Indicates why foreground work stopped.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.
    Agents SHOULD include this when the idle transition ends foreground work.
    """
    usage: Optional[Usage] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Token usage for completed foreground work.

    Optional. Omitted or `null` both mean the agent is not reporting token
    usage for this state update.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("stop_reason", "usage", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class RequiresActionStateUpdate(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class RunningState(RunningStateUpdate):
    state: Literal["running"] = "running"


class IdleState(IdleStateUpdate):
    state: Literal["idle"] = "idle"


class RequiresActionState(RequiresActionStateUpdate):
    state: Literal["requires_action"] = "requires_action"


class OtherState(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    state: str
    """
    Custom or future session state.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class StateUpdate(RootModel[Union[RunningState, IdleState, RequiresActionState, OtherState]]):
    root: Union[RunningState, IdleState, RequiresActionState, OtherState]
    """
    The state of the agent's foreground work has changed.

    Background activity can continue and emit other `session/update` notifications
    while `idle`. Those notifications do not change this state.
    """


class TerminalOutput(BaseModel):
    data: Annotated[str, Field(json_schema_extra={"contentEncoding": "base64"})]
    """
    Base64-encoded replacement terminal output bytes.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys. This metadata is scoped to the replacement snapshot. Omitted
    and `null` are equivalent and mean no snapshot metadata was provided.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class TerminalExitStatus(BaseModel):
    exit_code: Annotated[Optional[int], Field(alias="exitCode", ge=0)] = None
    """
    Process exit code, when known. Omitted and `null` are equivalent.
    """
    signal: Optional[str] = None
    """
    Signal that terminated the process, when known.

    Agents should use the conventional platform signal name. POSIX examples
    include `SIGTERM`, `SIGKILL`, and `SIGINT`. Other platforms may use a
    platform-specific name. Omitted and `null` are equivalent.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys. This metadata is scoped to the exit information. Omitted
    and `null` are equivalent and mean no exit metadata was provided.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("exit_code", "signal", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class TerminalUpdate(BaseModel):
    terminal_id: Annotated[str, Field(alias="terminalId")]
    """
    Unique identifier for this terminal within the session.
    """
    command: Optional[str] = None
    """
    The command being run.
    """
    cwd: Optional[str] = None
    """
    The absolute working directory of the command.
    """
    output: Optional[TerminalOutput] = None
    """
    An authoritative replacement snapshot of terminal output bytes.
    """
    exit_status: Annotated[Optional[TerminalExitStatus], Field(alias="exitStatus")] = None
    """
    Exit information. A concrete object marks the terminal as exited.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Omitted means no metadata update; `null` is an
    explicit clear signal. Implementations MUST NOT make assumptions about values at these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("command", "cwd", "exit_status", "output", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class TerminalOutputChunk(BaseModel):
    terminal_id: Annotated[str, Field(alias="terminalId")]
    """
    The terminal receiving these bytes.
    """
    data: Annotated[str, Field(json_schema_extra={"contentEncoding": "base64"})]
    """
    Independently base64-encoded terminal output bytes.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys. This field is chunk-scoped. Omitted and `null` are
    equivalent and mean no chunk metadata was provided.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class PlanFile(BaseModel):
    plan_id: Annotated[str, Field(alias="planId")]
    """
    The plan ID to update.
    """
    uri: AnyUrl
    """
    The URI of the file containing the plan.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class OtherAvailableCommandInput(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: str
    """
    Custom or future command input type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class TextCommandInput(BaseModel):
    hint: str
    """
    A hint to display when the input hasn't been provided yet
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionInfoUpdateBase(BaseModel):
    title: Optional[str] = None
    """
    Human-readable title for the session. Set to null to clear.
    """
    updated_at: Annotated[Optional[AwareDatetime], Field(alias="updatedAt")] = None
    """
    RFC 3339 timestamp of last activity. Set to null to clear.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Omitted means no metadata update; `null` is an
    explicit clear signal. Implementations MUST NOT make assumptions about values at these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class Cost(BaseModel):
    amount: float
    """
    Total cumulative cost for session.
    """
    currency: Annotated[str, Field(pattern="^[A-Z]{3}$")]
    """
    ISO 4217 currency code (e.g., "USD", "EUR").
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class UsageUpdateBase(BaseModel):
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


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

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("params", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class TerminalAuthCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ElicitationFormCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ElicitationUrlCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesJumpCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesRenameCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesSearchAndReplaceCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class LoginAuthRequest(BaseModel):
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ListProvidersRequest(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SetProviderRequest(BaseModel):
    provider_id: Annotated[str, Field(alias="providerId")]
    """
    Provider ID to configure.
    """
    api_type: Annotated[
        Union[Literal["anthropic"], Literal["openai"], Literal["azure"], Literal["vertex"], Literal["bedrock"], str],
        Field(alias="apiType"),
    ]
    """
    Protocol type for this provider.
    """
    base_url: Annotated[AnyUrl, Field(alias="baseUrl")]
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class LogoutAuthRequest(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class OtherMcpServer(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: str
    """
    Custom or future MCP server transport type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class McpServerHttp(BaseModel):
    name: str
    """
    Human-readable name identifying this MCP server.
    """
    url: AnyUrl
    """
    URL to the MCP server.
    """
    headers: Optional[List[HttpHeader]] = None
    """
    HTTP headers to set when making requests to the MCP server.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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
    args: Optional[List[str]] = None
    """
    Command-line arguments to pass to the MCP server.
    """
    env: Optional[List[EnvVariable]] = None
    """
    Environment variables to set when launching the MCP server.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class OtherReplayFrom(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: str
    """
    Custom or future replay cursor type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ReplayFromStart(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SetSessionConfigOptionIdRequest(BaseModel):
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    value: str
    """
    The value ID.
    """
    type: Literal["id"] = "id"


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    value: bool
    """
    The boolean value.
    """
    type: Literal["boolean"] = "boolean"


class SetSessionConfigOptionOtherRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    type: str
    """
    Custom or future session configuration option value type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """
    value: Any
    """
    Raw value payload for the custom or future value type.
    """


class WorkspaceFolder(BaseModel):
    uri: AnyUrl
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesRecentFile(BaseModel):
    uri: AnyUrl
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesEditHistoryEntry(BaseModel):
    uri: AnyUrl
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesUserAction(BaseModel):
    action: str
    """
    The kind of action (e.g., "insertChar", "cursorMovement").
    """
    uri: AnyUrl
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CancelledPermissionOutcome(BaseModel):
    outcome: Literal["cancelled"] = "cancelled"


class OtherPermissionOutcome(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    outcome: str
    """
    Custom or future permission outcome.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class DeclineElicitationResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    action: Literal["decline"] = "decline"


class CancelElicitationResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    action: Literal["cancel"] = "cancel"


class OtherElicitationResponse(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    action: str
    """
    Custom or future elicitation action.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class ElicitationContentValue(RootModel[Union[str, int, float, bool, List[str]]]):
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class DisconnectMcpResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CancelSessionNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to cancel operations for.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class DidOpenDocumentNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    uri: AnyUrl
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class DidCloseDocumentNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    uri: AnyUrl
    """
    The URI of the closed document.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class DidSaveDocumentNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    uri: AnyUrl
    """
    The URI of the saved document.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class AcceptNesNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    suggestion_id: Annotated[str, Field(alias="suggestionId")]
    """
    The ID of the accepted suggestion.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CommandPermissionSubjectVariant(CommandPermissionSubject):
    type: Literal["command"] = "command"


class TerminalToolCallContent(Terminal):
    type: Literal["terminal"] = "terminal"


class Annotations(BaseModel):
    audience: Optional[List[Union[Literal["assistant"], Literal["user"], str]]] = None
    """
    Intended recipients for this content, such as the user or assistant.
    """
    last_modified: Annotated[Optional[AwareDatetime], Field(alias="lastModified")] = None
    """
    Timestamp indicating when the underlying resource was last modified.

    Must be an RFC 3339 formatted string (e.g., "2025-01-12T15:00:58Z").
    """
    priority: Annotated[Optional[float], Field(ge=0.0, le=1.0)] = None
    """
    Relative importance of this content when clients choose what to surface.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("last_modified", "priority", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("audience", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class TextContent(BaseModel):
    text: str
    """
    Text payload carried by this content block.
    """
    annotations: Optional[Annotations] = None
    """
    Optional annotations that help clients decide how to display or route this content.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("annotations", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ImageContent(BaseModel):
    data: Annotated[str, Field(json_schema_extra={"contentEncoding": "base64"})]
    """
    Base64-encoded media payload.
    """
    mime_type: Annotated[str, Field(alias="mimeType")]
    """
    MIME type describing the encoded media payload.
    """
    uri: Optional[AnyUrl] = None
    """
    URI associated with this resource or media payload.
    """
    annotations: Optional[Annotations] = None
    """
    Optional annotations that help clients decide how to display or route this content.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("annotations", "uri", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class AudioContent(BaseModel):
    data: Annotated[str, Field(json_schema_extra={"contentEncoding": "base64"})]
    """
    Base64-encoded media payload.
    """
    mime_type: Annotated[str, Field(alias="mimeType")]
    """
    MIME type describing the encoded media payload.
    """
    annotations: Optional[Annotations] = None
    """
    Optional annotations that help clients decide how to display or route this content.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("annotations", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class Icon(BaseModel):
    src: AnyUrl
    """
    A standard URI pointing to an icon resource.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    Optional MIME type override if the source MIME type is missing or generic.
    """
    sizes: Optional[List[str]] = None
    """
    Optional array of strings that specify sizes at which the icon can be used.
    Each string should be in `WxH` format (e.g., `"48x48"`, `"96x96"`) or
    `"any"` for scalable formats like SVG.

    If not provided, the client should assume that the icon can be used at any size.
    """
    theme: Optional[Union[Literal["light"], Literal["dark"], str]] = None
    """
    Optional theme this icon is designed for.
    """

    @field_validator("mime_type", "theme", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("sizes", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class ResourceLink(BaseModel):
    name: str
    """
    Human-readable name shown for this protocol object.
    """
    uri: AnyUrl
    """
    URI associated with this resource or media payload.
    """
    title: Optional[str] = None
    """
    Optional display title for end-user UI.
    """
    description: Optional[str] = None
    """
    Optional human-readable details shown with this protocol object.
    """
    icons: Optional[List[Icon]] = None
    """
    Optional set of sized icons that the client can display in a user interface.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type describing the encoded media payload.
    """
    size: Optional[int] = None
    """
    Optional size of the linked resource in bytes, if known.
    """
    annotations: Optional[Annotations] = None
    """
    Optional annotations that help clients decide how to display or route this content.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("annotations", "description", "mime_type", "size", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("icons", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class EmbeddedResource(BaseModel):
    resource: Union[TextResourceContents, BlobResourceContents]
    """
    Embedded resource payload, either text or binary data.
    """
    annotations: Optional[Annotations] = None
    """
    Optional annotations that help clients decide how to display or route this content.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("annotations", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class AddDiffChange(DiffPathChange):
    file_type: Annotated[
        Optional[Union[Literal["text"], Literal["binary"], Literal["directory"], Literal["symlink"], str]],
        Field(alias="fileType"),
    ] = None
    """
    File content kind.

    Omitted or `null` means the content kind is unknown.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type of the file contents.

    Omitted or `null` means the MIME type is unknown.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    operation: Literal["add"] = "add"


class DeleteDiffChange(DiffPathChange):
    file_type: Annotated[
        Optional[Union[Literal["text"], Literal["binary"], Literal["directory"], Literal["symlink"], str]],
        Field(alias="fileType"),
    ] = None
    """
    File content kind.

    Omitted or `null` means the content kind is unknown.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type of the file contents.

    Omitted or `null` means the MIME type is unknown.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    operation: Literal["delete"] = "delete"


class ModifyDiffChange(DiffPathChange):
    file_type: Annotated[
        Optional[Union[Literal["text"], Literal["binary"], Literal["directory"], Literal["symlink"], str]],
        Field(alias="fileType"),
    ] = None
    """
    File content kind.

    Omitted or `null` means the content kind is unknown.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type of the file contents.

    Omitted or `null` means the MIME type is unknown.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    operation: Literal["modify"] = "modify"


class MoveDiffChange(DiffPathPairChange):
    file_type: Annotated[
        Optional[Union[Literal["text"], Literal["binary"], Literal["directory"], Literal["symlink"], str]],
        Field(alias="fileType"),
    ] = None
    """
    File content kind.

    Omitted or `null` means the content kind is unknown.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type of the file contents.

    Omitted or `null` means the MIME type is unknown.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    operation: Literal["move"] = "move"


class CopyDiffChange(DiffPathPairChange):
    file_type: Annotated[
        Optional[Union[Literal["text"], Literal["binary"], Literal["directory"], Literal["symlink"], str]],
        Field(alias="fileType"),
    ] = None
    """
    File content kind.

    Omitted or `null` means the content kind is unknown.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type of the file contents.

    Omitted or `null` means the MIME type is unknown.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    operation: Literal["copy"] = "copy"


class OtherDiffChange(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    file_type: Annotated[
        Optional[Union[Literal["text"], Literal["binary"], Literal["directory"], Literal["symlink"], str]],
        Field(alias="fileType"),
    ] = None
    """
    File content kind.

    Omitted or `null` means the content kind is unknown.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type of the file contents.

    Omitted or `null` means the MIME type is unknown.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    operation: str
    """
    Custom or future file operation.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class DiffPatch(BaseModel):
    format: Union[Literal["git_patch"], str]
    """
    Patch format. The only ACP-defined value is `git_patch`.
    """
    text: str
    """
    Patch text in the format named by `format`.
    """


class Diff(BaseModel):
    changes: List[
        Union[AddDiffChange, DeleteDiffChange, ModifyDiffChange, MoveDiffChange, CopyDiffChange, OtherDiffChange]
    ]
    """
    Structured file changes described by this diff.

    Clients can use this field without parsing patch text to determine affected paths.
    """
    patch: Optional[DiffPatch] = None
    """
    Renderable patch text for some or all of the structured changes.

    Agents SHOULD provide patch text whenever feasible. Omitted or `null`
    means no renderable patch text was provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("patch", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("changes", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class PermissionOption(BaseModel):
    option_id: Annotated[str, Field(alias="optionId")]
    """
    Unique identifier for this permission option.
    """
    name: str
    """
    Human-readable label to display to the user.
    """
    kind: Union[Literal["allow_once"], Literal["allow_always"], Literal["reject_once"], Literal["reject_always"], str]
    """
    Hint about the nature of this permission option.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CreateUrlSessionElicitationRequestBase(ElicitationSessionScope):
    elicitation_id: Annotated[str, Field(alias="elicitationId")]
    """
    The unique identifier for this elicitation.
    """
    url: AnyUrl
    """
    The URL to direct the user to.
    """


class CreateUrlRequestElicitationRequestBase(ElicitationRequestScope):
    elicitation_id: Annotated[str, Field(alias="elicitationId")]
    """
    The unique identifier for this elicitation.
    """
    url: AnyUrl
    """
    The URL to direct the user to.
    """


class CreateUrlSessionElicitationRequest(CreateUrlSessionElicitationRequestBase, CreateUrlElicitationRequestBase):
    message: str
    """
    A human-readable message describing what input is needed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    mode: Literal["url"] = "url"


class CreateUrlRequestElicitationRequest(CreateUrlRequestElicitationRequestBase, CreateUrlElicitationRequestBase):
    message: str
    """
    A human-readable message describing what input is needed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    mode: Literal["url"] = "url"


class CreateOtherSessionElicitationRequest(ElicitationSessionScope):
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

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    mode: str
    """
    Custom or future elicitation mode.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class CreateOtherRequestElicitationRequest(ElicitationRequestScope):
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

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    mode: str
    """
    Custom or future elicitation mode.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class ElicitationStringPropertySchema(StringPropertySchema):
    type: Literal["string"] = "string"


class ElicitationNumberPropertySchema(NumberPropertySchema):
    type: Literal["number"] = "number"


class ElicitationIntegerPropertySchema(IntegerPropertySchema):
    type: Literal["integer"] = "integer"


class ElicitationBooleanPropertySchema(BooleanPropertySchema):
    type: Literal["boolean"] = "boolean"


class StringMultiSelectItems(StringMultiSelectItemsBase):
    type: Literal["string"] = "string"


class MultiSelectPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional title for the property.

    Optional. Omitted and `null` are equivalent and mean no title is provided.
    """
    description: Optional[str] = None
    """
    Human-readable description.

    Optional. Omitted and `null` are equivalent and mean no description is provided.
    """
    min_items: Annotated[Optional[int], Field(alias="minItems", ge=0)] = None
    """
    Minimum number of items to select.

    Optional. Omitted and `null` are equivalent and mean there is no minimum selection count.
    """
    max_items: Annotated[Optional[int], Field(alias="maxItems", ge=0)] = None
    """
    Maximum number of items to select.

    Optional. Omitted and `null` are equivalent and mean there is no maximum selection count.
    """
    items: Union[StringMultiSelectItems, OtherMultiSelectItems, TitledMultiSelectItems]
    """
    The items definition describing allowed values.
    """
    default: Optional[List[str]] = None
    """
    Default selected values.

    Optional. Omitted and `null` are equivalent and mean no default selections are provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("description", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("default", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class PromptCapabilities(BaseModel):
    image: Optional[PromptImageCapabilities] = None
    """
    Agent supports [`ContentBlock::Image`].

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports image content in prompts.
    """
    audio: Optional[PromptAudioCapabilities] = None
    """
    Agent supports [`ContentBlock::Audio`].

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports audio content in prompts.
    """
    embedded_context: Annotated[Optional[PromptEmbeddedContextCapabilities], Field(alias="embeddedContext")] = None
    """
    Agent supports embedded context in `session/prompt` requests.

    When enabled, the Client is allowed to include [`ContentBlock::Resource`]
    in prompt requests for pieces of context that are referenced in the message.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports embedded context in prompts.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("audio", "embedded_context", "image", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class McpCapabilities(BaseModel):
    stdio: Optional[McpStdioCapabilities] = None
    """
    Agent supports [`McpServer::Stdio`].

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports stdio MCP server transports.
    """
    http: Optional[McpHttpCapabilities] = None
    """
    Agent supports [`McpServer::Http`].

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports HTTP MCP server transports.
    """
    acp: Optional[McpAcpCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Agent supports [`McpServer::Acp`].

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports ACP MCP server transports.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("acp", "http", "stdio", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class NesDocumentDidChangeCapabilities(BaseModel):
    sync_kind: Annotated[Literal["full", "incremental"], Field(alias="syncKind")]
    """
    The sync kind the agent wants: `"full"` or `"incremental"`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator(
        "diagnostics", "edit_history", "open_files", "recent_files", "related_snippets", "user_actions", mode="wrap"
    )
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class TerminalAuthMethod(AuthMethodTerminal):
    type: Literal["terminal"] = "terminal"


class AgentAuthMethod(AuthMethodAgent):
    type: Literal["agent"] = "agent"


class OtherAuthMethod(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: str
    """
    Custom or future authentication method type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """
    method_id: Annotated[str, Field(alias="methodId")]
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ProviderInfo(BaseModel):
    provider_id: Annotated[str, Field(alias="providerId")]
    """
    Provider identifier, for example "main" or "openai".
    """
    supported: List[
        Union[Literal["anthropic"], Literal["openai"], Literal["azure"], Literal["vertex"], Literal["bedrock"], str]
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("supported", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class BooleanSessionConfigOption(SessionConfigBoolean):
    config_id: Annotated[str, Field(alias="configId")]
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
        Union[Literal["mode"], Literal["model"], Literal["model_config"], Literal["thought_level"], str]
    ] = None
    """
    Optional semantic category for this option (UX only).
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    type: Literal["boolean"] = "boolean"


class OtherSessionConfigOption(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    config_id: Annotated[str, Field(alias="configId")]
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
        Union[Literal["mode"], Literal["model"], Literal["model_config"], Literal["thought_level"], str]
    ] = None
    """
    Optional semantic category for this option (UX only).
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    type: str
    """
    Custom or future session configuration option type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class SessionConfigSelectGroup(BaseModel):
    group_id: Annotated[str, Field(alias="groupId")]
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class ListSessionsResponse(BaseModel):
    sessions: List[SessionInfo]
    """
    Array of session information objects.
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("next_cursor", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("sessions", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class NesJumpSuggestionVariant(NesJumpSuggestion):
    kind: Literal["jump"] = "jump"


class NesRenameSuggestionVariant(NesRenameSuggestion):
    kind: Literal["rename"] = "rename"


class NesSearchAndReplaceSuggestionVariant(NesSearchAndReplaceSuggestion):
    kind: Literal["searchAndReplace"] = "searchAndReplace"


class OtherNesSuggestion(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    kind: str
    """
    Custom or future NES suggestion kind.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """
    suggestion_id: Annotated[str, Field(alias="suggestionId")]
    """
    Unique identifier for accept/reject tracking.
    """


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class RunningSessionStateUpdateBase(RunningStateUpdate):
    state: Literal["running"] = "running"


class IdleSessionStateUpdateBase(IdleStateUpdate):
    state: Literal["idle"] = "idle"


class RequiresActionSessionStateUpdateBase(RequiresActionStateUpdate):
    state: Literal["requires_action"] = "requires_action"


class RunningSessionStateUpdate(RunningSessionStateUpdateBase, SessionStateUpdateBase):
    session_update: Annotated[Literal["state_update"], Field(alias="sessionUpdate")] = "state_update"


class IdleSessionStateUpdate(IdleSessionStateUpdateBase, SessionStateUpdateBase):
    session_update: Annotated[Literal["state_update"], Field(alias="sessionUpdate")] = "state_update"


class RequiresActionSessionStateUpdate(RequiresActionSessionStateUpdateBase, SessionStateUpdateBase):
    session_update: Annotated[Literal["state_update"], Field(alias="sessionUpdate")] = "state_update"


class SessionTerminalUpdate(TerminalUpdate):
    session_update: Annotated[Literal["terminal_update"], Field(alias="sessionUpdate")] = "terminal_update"


class SessionTerminalOutputChunk(TerminalOutputChunk):
    session_update: Annotated[Literal["terminal_output_chunk"], Field(alias="sessionUpdate")] = "terminal_output_chunk"


class SessionPlanRemovedUpdate(PlanRemoved):
    session_update: Annotated[Literal["plan_removed"], Field(alias="sessionUpdate")] = "plan_removed"


class SessionInfoUpdate(SessionInfoUpdateBase):
    session_update: Annotated[Literal["session_info_update"], Field(alias="sessionUpdate")] = "session_info_update"

    @field_validator("title", "updated_at", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class UsageUpdate(UsageUpdateBase):
    session_update: Annotated[Literal["usage_update"], Field(alias="sessionUpdate")] = "usage_update"

    @field_validator("cost", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class PlanUpdateFile(PlanFile):
    type: Literal["file"] = "file"


class PlanUpdateMarkdown(PlanMarkdown):
    type: Literal["markdown"] = "markdown"


class OtherPlanUpdateContent(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: str
    """
    Custom or future plan update content type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """
    plan_id: Annotated[str, Field(alias="planId")]
    """
    The plan ID to update.
    """


class PlanEntry(BaseModel):
    content: str
    """
    Human-readable description of what this task aims to accomplish.
    """
    priority: Union[Literal["high"], Literal["medium"], Literal["low"], str]
    """
    The relative importance of this task.
    Used to indicate which tasks are most critical to the overall goal.
    """
    status: Union[Literal["pending"], Literal["in_progress"], Literal["completed"], Literal["cancelled"], str]
    """
    Current execution status of this task.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("entries", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class TextAvailableCommandInput(TextCommandInput):
    type: Literal["text"] = "text"


class AuthCapabilities(BaseModel):
    terminal: Optional[TerminalAuthCapabilities] = None
    """
    Whether the client supports `terminal` authentication methods.

    Optional. Omitted or `null` both mean the client does not advertise support.
    The client should supply `{}` only when it can reproduce the configured
    agent invocation in an interactive terminal. Supplying `{}` means the
    agent may include `terminal` entries in its authentication methods.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("terminal", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ElicitationCapabilities(BaseModel):
    form: Optional[ElicitationFormCapabilities] = None
    """
    Whether the client supports form-based elicitation.

    Optional. Omitted and `null` are equivalent and mean form support is not advertised.
    Supplying `{}` explicitly advertises form support.
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

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("form", "url", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("jump", "rename", "search_and_replace", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class HttpMcpServer(McpServerHttp):
    type: Literal["http"] = "http"


class AcpMcpServer(McpServerAcp):
    type: Literal["acp"] = "acp"


class StdioMcpServer(McpServerStdio):
    type: Literal["stdio"] = "stdio"


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
        Optional[List[Union[HttpMcpServer, AcpMcpServer, StdioMcpServer, OtherMcpServer]]], Field(alias="mcpServers")
    ] = None
    """
    List of MCP servers to connect to for this session.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("additional_directories", "mcp_servers", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class ReplayFromStartVariant(ReplayFromStart):
    type: Literal["start"] = "start"


class StartNesRequest(BaseModel):
    workspace_uri: Annotated[Optional[AnyUrl], Field(alias="workspaceUri")] = None
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("repository", "workspace_uri", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class NesRelatedSnippet(BaseModel):
    uri: AnyUrl
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesOpenFile(BaseModel):
    uri: AnyUrl
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("last_focused_ms", "visible_range", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class NesDiagnostic(BaseModel):
    uri: AnyUrl
    """
    The URI of the file containing the diagnostic.
    """
    range: Range
    """
    The range of the diagnostic.
    """
    severity: Union[Literal["error"], Literal["warning"], Literal["information"], Literal["hint"], str]
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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


class SelectedPermissionOutcomeVariant(SelectedPermissionOutcome):
    outcome: Literal["selected"] = "selected"


class AcceptElicitationResponse(ElicitationAcceptAction):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    action: Literal["accept"] = "accept"


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class DidFocusDocumentNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    uri: AnyUrl
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class RejectNesNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    suggestion_id: Annotated[str, Field(alias="suggestionId")]
    """
    The ID of the rejected suggestion.
    """
    reason: Optional[Union[Literal["rejected"], Literal["ignored"], Literal["replaced"], Literal["cancelled"], str]] = (
        None
    )
    """
    The reason for rejection.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("reason", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ProtocolLevelNotification(BaseModel):
    method: str
    """
    The notification method name.
    """
    params: Optional[CancelRequestNotification] = None
    """
    Method-specific notification parameters.
    """


class DiffToolCallContent(Diff):
    type: Literal["diff"] = "diff"


class TextContentBlock(TextContent):
    type: Literal["text"] = "text"


class ImageContentBlock(ImageContent):
    type: Literal["image"] = "image"


class AudioContentBlock(AudioContent):
    type: Literal["audio"] = "audio"


class ResourceContentBlock(ResourceLink):
    type: Literal["resource_link"] = "resource_link"


class EmbeddedResourceContentBlock(EmbeddedResource):
    type: Literal["resource"] = "resource"


class Content(BaseModel):
    content: Union[
        TextContentBlock,
        ImageContentBlock,
        AudioContentBlock,
        ResourceContentBlock,
        EmbeddedResourceContentBlock,
        OtherContentBlock,
    ]
    """
    The actual content block.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ElicitationMultiSelectPropertySchema(MultiSelectPropertySchema):
    type: Literal["array"] = "array"


class AgentErrorMessage(BaseModel):
    id: Optional[Union[int, str]]
    """
    The id of the request this response answers.
    """
    error: Error
    """
    Method-specific error data.
    """


class SessionCapabilities(BaseModel):
    prompt: Optional[PromptCapabilities] = None
    """
    Prompt capabilities supported by the agent in `session/prompt` requests.

    Optional. Omitted or `null` both mean the agent does not advertise any
    prompt extensions beyond the baseline text and resource-link content
    required by `session/prompt`.
    """
    mcp: Optional[McpCapabilities] = None
    """
    MCP capabilities supported by the agent for session lifecycle requests.

    Optional. Omitted or `null` both mean the agent does not advertise MCP
    server transport support for sessions.
    """
    delete: Optional[SessionDeleteCapabilities] = None
    """
    Whether the agent supports `session/delete`.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports deleting sessions from `session/list`.
    """
    additional_directories: Annotated[
        Optional[SessionAdditionalDirectoriesCapabilities], Field(alias="additionalDirectories")
    ] = None
    """
    Whether the agent supports `additionalDirectories` on supported session lifecycle requests.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports `additionalDirectories` on
    supported session lifecycle requests.

    Agents may return `SessionInfo.additionalDirectories` to report the
    complete ordered additional-root list associated with a listed session.
    """
    fork: Optional[SessionForkCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the agent supports `session/fork`.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports forking sessions.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("additional_directories", "delete", "fork", "mcp", "prompt", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("did_change", "did_close", "did_focus", "did_open", "did_save", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class NesEditSuggestion(BaseModel):
    suggestion_id: Annotated[str, Field(alias="suggestionId")]
    """
    Unique identifier for accept/reject tracking.
    """
    uri: AnyUrl
    """
    The URI of the file to edit.
    """
    edits: Annotated[List[NesTextEdit], Field(min_length=1)]
    """
    The text edits to apply. Must contain at least one edit.
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("cursor_position", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ContentChunk(BaseModel):
    message_id: Annotated[str, Field(alias="messageId")]
    """
    A unique identifier for the message this chunk belongs to.

    All chunks belonging to the same message share the same `messageId`.
    A change in `messageId` indicates a new message has started.
    """
    content: Union[
        TextContentBlock,
        ImageContentBlock,
        AudioContentBlock,
        ResourceContentBlock,
        EmbeddedResourceContentBlock,
        OtherContentBlock,
    ]
    """
    A single item of content
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys. This field is chunk-scoped.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class UserMessage(BaseModel):
    message_id: Annotated[str, Field(alias="messageId")]
    """
    A unique identifier for the message.
    """
    content: Optional[
        List[
            Union[
                TextContentBlock,
                ImageContentBlock,
                AudioContentBlock,
                ResourceContentBlock,
                EmbeddedResourceContentBlock,
                OtherContentBlock,
            ]
        ]
    ] = None
    """
    Complete replacement content for this message.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys. Omitted means no metadata update; `null` is an explicit clear signal.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("content", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class AgentMessage(BaseModel):
    message_id: Annotated[str, Field(alias="messageId")]
    """
    A unique identifier for the message.
    """
    content: Optional[
        List[
            Union[
                TextContentBlock,
                ImageContentBlock,
                AudioContentBlock,
                ResourceContentBlock,
                EmbeddedResourceContentBlock,
                OtherContentBlock,
            ]
        ]
    ] = None
    """
    Complete replacement content for this message.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys. Omitted means no metadata update; `null` is an explicit clear signal.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("content", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class AgentThought(BaseModel):
    message_id: Annotated[str, Field(alias="messageId")]
    """
    A unique identifier for the thought message.
    """
    content: Optional[
        List[
            Union[
                TextContentBlock,
                ImageContentBlock,
                AudioContentBlock,
                ResourceContentBlock,
                EmbeddedResourceContentBlock,
                OtherContentBlock,
            ]
        ]
    ] = None
    """
    Complete replacement content for this thought message.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys. Omitted means no metadata update; `null` is an explicit clear signal.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("content", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class PlanUpdateItems(PlanItems):
    type: Literal["items"] = "items"


class PlanUpdate(BaseModel):
    plan: Union[PlanUpdateItems, PlanUpdateFile, PlanUpdateMarkdown, OtherPlanUpdateContent]
    """
    The updated plan content.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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
    input: Optional[Union[TextAvailableCommandInput, OtherAvailableCommandInput]] = None
    """
    Input for the command if required
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("input", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class AvailableCommandsUpdateBase(BaseModel):
    available_commands: Annotated[List[AvailableCommand], Field(alias="availableCommands")]
    """
    Commands the agent can execute.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CompactionUpdate(BaseModel):
    compaction_id: Annotated[str, Field(alias="compactionId")]
    """
    The Agent-owned ID of this compaction, unique within the session.
    """
    status: Union[Literal["in_progress"], Literal["completed"], Literal["failed"], Literal["cancelled"], str]
    """
    Current lifecycle status.
    """
    summary: Optional[
        List[
            Union[
                TextContentBlock,
                ImageContentBlock,
                AudioContentBlock,
                ResourceContentBlock,
                EmbeddedResourceContentBlock,
                OtherContentBlock,
            ]
        ]
    ] = None
    """
    Complete replacement user-displayable summary retained by the compaction.
    """
    error: Optional[str] = None
    """
    Human-readable description of why the compaction failed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    Extensible metadata patch for this compaction.
    """

    @field_validator("error", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("summary", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class CompactionSummaryChunk(BaseModel):
    compaction_id: Annotated[str, Field(alias="compactionId")]
    """
    ID of the compaction whose summary receives this content.
    """
    content: Union[
        TextContentBlock,
        ImageContentBlock,
        AudioContentBlock,
        ResourceContentBlock,
        EmbeddedResourceContentBlock,
        OtherContentBlock,
    ]
    """
    One content block to append.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    Metadata scoped to this chunk. Omission and `null` both mean absent.
    """


class ClientCapabilities(BaseModel):
    auth: Optional[AuthCapabilities] = None
    """
    Authentication capabilities supported by the client.
    Determines which authentication method types the agent may include
    in its `InitializeResponse`.

    Optional. Omitted or `null` both mean the client does not advertise any
    authentication-method extensions.
    """
    elicitation: Optional[ElicitationCapabilities] = None
    """
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
    position_encodings: Annotated[
        Optional[List[Literal["utf-16", "utf-32", "utf-8"]]], Field(alias="positionEncodings")
    ] = None
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("auth", "elicitation", "nes", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("position_encodings", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class NewSessionRequest(BaseModel):
    cwd: str
    """
    The working directory for this session. Must be an absolute path.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = None
    """
    Additional workspace roots for this session. Each path must be absolute.

    These expand the session's workspace scope without changing `cwd`, which
    remains the base for relative paths. When omitted or empty, no
    additional roots are activated for the new session.
    """
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, AcpMcpServer, StdioMcpServer, OtherMcpServer]]], Field(alias="mcpServers")
    ] = None
    """
    List of MCP (Model Context Protocol) servers the agent should connect to.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("additional_directories", "mcp_servers", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


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
        Optional[List[Union[HttpMcpServer, AcpMcpServer, StdioMcpServer, OtherMcpServer]]], Field(alias="mcpServers")
    ] = None
    """
    List of MCP servers to connect to for this session.
    """
    replay_from: Annotated[Optional[Union[ReplayFromStartVariant, OtherReplayFrom]], Field(alias="replayFrom")] = None
    """
    Inclusive cursor describing where conversation replay should begin.

    Optional. Omitted or `null` both mean the Agent should resume without
    replaying previous conversation history. Replay cursors are inclusive:
    replay includes the position identified by the cursor. Supplying
    `{ "type": "start" }` means the Agent should replay the whole
    conversation before responding.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("replay_from", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("additional_directories", "mcp_servers", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class PromptRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to send this user message to
    """
    prompt: List[
        Union[
            TextContentBlock,
            ImageContentBlock,
            AudioContentBlock,
            ResourceContentBlock,
            EmbeddedResourceContentBlock,
            OtherContentBlock,
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class RequestPermissionResponse(BaseModel):
    outcome: Union[CancelledPermissionOutcome, SelectedPermissionOutcomeVariant, OtherPermissionOutcome]
    """
    The user's decision on the permission request.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class DidChangeDocumentNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this notification.
    """
    uri: AnyUrl
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("content_changes", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class ContentToolCallContent(Content):
    type: Literal["content"] = "content"


class ElicitationSchema(BaseModel):
    type: Optional[Literal["object"]] = "object"
    """
    Type discriminator. Always `"object"`.
    """
    title: Optional[str] = None
    """
    Optional title for the schema.

    Optional. Omitted and `null` are equivalent and mean no title is provided.
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

    Optional. Omitted and `null` are equivalent and mean no property names are required.
    """
    description: Optional[str] = None
    """
    Optional description of what this schema represents.

    Optional. Omitted and `null` are equivalent and mean no schema description is provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("description", "title", "type", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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
    root: Union[ElicitationFormSessionMode, ElicitationFormRequestMode]
    """
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("document", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SelectSessionConfigOption(SessionConfigSelect):
    config_id: Annotated[str, Field(alias="configId")]
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
        Union[Literal["mode"], Literal["model"], Literal["model_config"], Literal["thought_level"], str]
    ] = None
    """
    Optional semantic category for this option (UX only).
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    type: Literal["select"] = "select"


class ForkSessionResponse(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    Unique identifier for the newly created forked session.
    """
    config_options: Annotated[
        Optional[List[Union[SelectSessionConfigOption, BooleanSessionConfigOption, OtherSessionConfigOption]]],
        Field(alias="configOptions"),
    ] = None
    """
    Initial session configuration options.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class ResumeSessionResponse(BaseModel):
    config_options: Annotated[
        Optional[List[Union[SelectSessionConfigOption, BooleanSessionConfigOption, OtherSessionConfigOption]]],
        Field(alias="configOptions"),
    ] = None
    """
    Initial session configuration options.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class SetSessionConfigOptionResponse(BaseModel):
    config_options: Annotated[
        List[Union[SelectSessionConfigOption, BooleanSessionConfigOption, OtherSessionConfigOption]],
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class NesEditSuggestionVariant(NesEditSuggestion):
    kind: Literal["edit"] = "edit"


class UserMessageChunk(ContentChunk):
    session_update: Annotated[Literal["user_message_chunk"], Field(alias="sessionUpdate")] = "user_message_chunk"


class UserMessageUpdate(UserMessage):
    session_update: Annotated[Literal["user_message"], Field(alias="sessionUpdate")] = "user_message"


class AgentMessageChunk(ContentChunk):
    session_update: Annotated[Literal["agent_message_chunk"], Field(alias="sessionUpdate")] = "agent_message_chunk"


class AgentMessageUpdate(AgentMessage):
    session_update: Annotated[Literal["agent_message"], Field(alias="sessionUpdate")] = "agent_message"


class AgentThoughtChunk(ContentChunk):
    session_update: Annotated[Literal["agent_thought_chunk"], Field(alias="sessionUpdate")] = "agent_thought_chunk"


class AgentThoughtUpdate(AgentThought):
    session_update: Annotated[Literal["agent_thought"], Field(alias="sessionUpdate")] = "agent_thought"


class SessionPlanUpdate(PlanUpdate):
    session_update: Annotated[Literal["plan_update"], Field(alias="sessionUpdate")] = "plan_update"


class AvailableCommandsUpdate(AvailableCommandsUpdateBase):
    session_update: Annotated[Literal["available_commands_update"], Field(alias="sessionUpdate")] = (
        "available_commands_update"
    )

    @field_validator("available_commands", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class SessionCompactionUpdate(CompactionUpdate):
    session_update: Annotated[Literal["compaction_update"], Field(alias="sessionUpdate")] = "compaction_update"


class SessionCompactionSummaryChunk(CompactionSummaryChunk):
    session_update: Annotated[Literal["compaction_summary_chunk"], Field(alias="sessionUpdate")] = (
        "compaction_summary_chunk"
    )


class ToolCallContentChunk(BaseModel):
    tool_call_id: Annotated[str, Field(alias="toolCallId")]
    """
    The ID of the tool call this content belongs to.
    """
    content: Union[ContentToolCallContent, DiffToolCallContent, TerminalToolCallContent, OtherToolCallContent]
    """
    A single item of content produced by the tool call.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys. This field is chunk-scoped.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ConfigOptionUpdateBase(BaseModel):
    config_options: Annotated[
        List[Union[SelectSessionConfigOption, BooleanSessionConfigOption, OtherSessionConfigOption]],
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class InitializeRequest(BaseModel):
    protocol_version: Annotated[int, Field(alias="protocolVersion", ge=0, le=65535)]
    """
    The latest protocol version supported by the client.
    """
    info: Implementation
    """
    Information about the implementation sending this initialize request.
    """
    capabilities: Annotated[Optional[ClientCapabilities], Field(validate_default=True)] = {}
    """
    Capabilities supported by the client.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("protocol_version", mode="before")
    @classmethod
    def coerce_protocol_version_validator(cls, v: Any, info: ValidationInfo) -> Any:
        return coerce_protocol_version(v, info)

    @field_validator("capabilities", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SuggestNesRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this request.
    """
    uri: AnyUrl
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
    trigger_kind: Annotated[
        Union[Literal["automatic"], Literal["diagnostic"], Literal["manual"], str], Field(alias="triggerKind")
    ]
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ClientResponseMessage(BaseModel):
    id: Optional[Union[int, str]]
    """
    The id of the request this response answers.
    """
    result: Union[
        RequestPermissionResponse,
        ConnectMcpResponse,
        DisconnectMcpResponse,
        Union[
            AcceptElicitationResponse, DeclineElicitationResponse, CancelElicitationResponse, OtherElicitationResponse
        ],
        Any,
    ]
    """
    Method-specific response data.
    """


class ClientResponse(RootModel[Union[ClientResponseMessage, ClientErrorMessage]]):
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
            CancelSessionNotification,
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
    Unique identifier for this tool call within the session.
    """
    name: Optional[str] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Programmatic name of the tool being invoked.

    This field is optional and has patch semantics. Omission means no
    change, `null` clears the name, and a string replaces it. For a tool
    call ID the client has not seen before, omission or `null` means that no
    tool name is available.
    """
    title: Optional[str] = None
    """
    Human-readable title describing what the tool is doing.
    """
    kind: Optional[
        Union[
            Literal["read"],
            Literal["edit"],
            Literal["delete"],
            Literal["move"],
            Literal["search"],
            Literal["execute"],
            Literal["think"],
            Literal["fetch"],
            Literal["switch_mode"],
            Literal["other"],
            str,
        ]
    ] = None
    """
    The category of tool being invoked.
    Helps clients choose appropriate icons and UI treatment.
    """
    status: Optional[
        Union[
            Literal["pending"],
            Literal["in_progress"],
            Literal["completed"],
            Literal["failed"],
            Literal["cancelled"],
            str,
        ]
    ] = None
    """
    Current execution status of the tool call.
    """
    content: Optional[
        List[Union[ContentToolCallContent, DiffToolCallContent, TerminalToolCallContent, OtherToolCallContent]]
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
    metadata to their interactions. Omitted means no metadata update; `null` is an
    explicit clear signal. Implementations MUST NOT make assumptions about values at these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("kind", "name", "raw_input", "raw_output", "status", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("content", "locations", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class ToolCallPermissionSubject(BaseModel):
    tool_call: Annotated[ToolCallUpdate, Field(alias="toolCall")]
    """
    Details about the tool call requiring permission.
    """


class CreateFormSessionElicitationRequestBase(ElicitationSessionScope):
    requested_schema: Annotated[ElicitationSchema, Field(alias="requestedSchema")]
    """
    A JSON Schema describing the form fields to present to the user.
    """


class CreateFormRequestElicitationRequestBase(ElicitationRequestScope):
    requested_schema: Annotated[ElicitationSchema, Field(alias="requestedSchema")]
    """
    A JSON Schema describing the form fields to present to the user.
    """


class CreateFormSessionElicitationRequest(CreateFormSessionElicitationRequestBase, CreateFormElicitationRequestBase):
    message: str
    """
    A human-readable message describing what input is needed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    mode: Literal["form"] = "form"


class CreateFormRequestElicitationRequest(CreateFormRequestElicitationRequestBase, CreateFormElicitationRequestBase):
    message: str
    """
    A human-readable message describing what input is needed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    mode: Literal["form"] = "form"


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("context", "events", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class NewSessionResponse(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    Unique identifier for the created session.

    Used in all subsequent requests for this conversation.
    """
    config_options: Annotated[
        Optional[List[Union[SelectSessionConfigOption, BooleanSessionConfigOption, OtherSessionConfigOption]]],
        Field(alias="configOptions"),
    ] = None
    """
    Initial session configuration options.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class SuggestNesResponse(BaseModel):
    suggestions: List[
        Union[
            NesEditSuggestionVariant,
            NesJumpSuggestionVariant,
            NesRenameSuggestionVariant,
            NesSearchAndReplaceSuggestionVariant,
            OtherNesSuggestion,
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ToolCallContentChunkUpdate(ToolCallContentChunk):
    session_update: Annotated[Literal["tool_call_content_chunk"], Field(alias="sessionUpdate")] = (
        "tool_call_content_chunk"
    )


class SessionToolCallUpdate(ToolCallUpdate):
    session_update: Annotated[Literal["tool_call_update"], Field(alias="sessionUpdate")] = "tool_call_update"


class ConfigOptionUpdate(ConfigOptionUpdateBase):
    session_update: Annotated[Literal["config_option_update"], Field(alias="sessionUpdate")] = "config_option_update"

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


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
            LoginAuthRequest,
            ListProvidersRequest,
            SetProviderRequest,
            DisableProviderRequest,
            LogoutAuthRequest,
            NewSessionRequest,
            ListSessionsRequest,
            DeleteSessionRequest,
            ForkSessionRequest,
            ResumeSessionRequest,
            CloseSessionRequest,
            PromptRequest,
            StartNesRequest,
            SuggestNesRequest,
            CloseNesRequest,
            MessageMcpRequest,
            Union[
                SetSessionConfigOptionIdRequest,
                SetSessionConfigOptionBooleanRequest,
                SetSessionConfigOptionOtherRequest,
            ],
            Any,
        ]
    ] = None
    """
    Method-specific request parameters.
    """


class ToolCallPermissionSubjectVariant(ToolCallPermissionSubject):
    type: Literal["tool_call"] = "tool_call"


class AgentCapabilities(BaseModel):
    session: Optional[SessionCapabilities] = None
    """
    Session capabilities supported by the agent.

    Optional. Omitted or `null` both mean the agent does not support the
    `session/*` method surface. Supplying `{}` means the agent supports the
    baseline session methods: `session/new`, `session/prompt`,
    `session/cancel`, and `session/update`.
    """
    auth: Optional[AgentAuthCapabilities] = None
    """
    Authentication-related extension capabilities supported by the agent.

    Optional. Omitted or `null` both mean the agent does not advertise any
    authentication-related extensions. This field does not advertise support
    for `auth/login` or `auth/logout`; those methods are advertised by a
    non-empty `authMethods` list in the `initialize` response.
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
    position_encoding: Annotated[Optional[Literal["utf-16", "utf-32", "utf-8"]], Field(alias="positionEncoding")] = None
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("auth", "nes", "position_encoding", "providers", "session", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class UpdateSessionNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session this update pertains to.
    """
    update: Union[
        UserMessageChunk,
        UserMessageUpdate,
        AgentMessageChunk,
        AgentMessageUpdate,
        AgentThoughtChunk,
        AgentThoughtUpdate,
        ToolCallContentChunkUpdate,
        SessionToolCallUpdate,
        SessionTerminalUpdate,
        SessionTerminalOutputChunk,
        SessionPlanUpdate,
        SessionPlanRemovedUpdate,
        AvailableCommandsUpdate,
        ConfigOptionUpdate,
        SessionInfoUpdate,
        UsageUpdate,
        SessionCompactionUpdate,
        SessionCompactionSummaryChunk,
        OtherSessionUpdate,
        Union[
            RunningSessionStateUpdate, IdleSessionStateUpdate, RequiresActionSessionStateUpdate, OtherSessionStateUpdate
        ],
    ]
    """
    The actual update content.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class RequestPermissionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this request.
    """
    title: str
    """
    Human-readable title for the permission prompt.

    This title is specific to the permission prompt and does not update any
    subject's displayed title.
    """
    description: Optional[str] = None
    """
    Optional human-readable explanation of why permission is needed.

    This text is specific to the permission prompt and does not update any
    subject's displayed content. Omitted or `null` both mean no separate
    permission description was provided.
    """
    subject: Optional[
        Union[ToolCallPermissionSubjectVariant, CommandPermissionSubjectVariant, OtherPermissionSubject]
    ] = None
    """
    Optional structured context about the operation requiring permission.

    Omitted or `null` both mean no structured subject was provided.
    """
    options: Annotated[List[PermissionOption], Field(min_length=1)]
    """
    Available permission options for the user to choose from.
    Must contain at least one option.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class InitializeResponse(BaseModel):
    protocol_version: Annotated[int, Field(alias="protocolVersion", ge=0, le=65535)]
    """
    The protocol version the client specified if supported by the agent,
    or the latest protocol version supported by the agent.

    The client should disconnect, if it doesn't support this version.
    """
    info: Implementation
    """
    Information about the implementation sending this initialize response.
    """
    capabilities: Annotated[Optional[AgentCapabilities], Field(validate_default=True)] = {}
    """
    Capabilities supported by the agent.
    """
    auth_methods: Annotated[
        Optional[List[Union[TerminalAuthMethod, AgentAuthMethod, OtherAuthMethod]]], Field(alias="authMethods")
    ] = None
    """
    Authentication methods supported by the agent.

    Optional. Omitted or empty means the agent does not advertise the
    authentication method surface. Supplying one or more valid methods means
    the agent MUST support both `auth/login` and `auth/logout`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("capabilities", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("auth_methods", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class AgentNotification(BaseModel):
    method: str
    """
    The notification method name.
    """
    params: Optional[Union[UpdateSessionNotification, CompleteElicitationNotification, MessageMcpNotification, Any]] = (
        None
    )
    """
    Method-specific notification parameters.
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
            RequestPermissionRequest,
            ConnectMcpRequest,
            MessageMcpRequest,
            DisconnectMcpRequest,
            Union[
                Union[CreateOtherSessionElicitationRequest, CreateOtherRequestElicitationRequest],
                Union[CreateFormSessionElicitationRequest, CreateFormRequestElicitationRequest],
                Union[CreateUrlSessionElicitationRequest, CreateUrlRequestElicitationRequest],
            ],
            Any,
        ]
    ] = None
    """
    Method-specific request parameters.
    """


class AgentResponseMessage(BaseModel):
    id: Optional[Union[int, str]]
    """
    The id of the request this response answers.
    """
    result: Union[
        InitializeResponse,
        LoginAuthResponse,
        ListProvidersResponse,
        SetProviderResponse,
        DisableProviderResponse,
        LogoutAuthResponse,
        NewSessionResponse,
        ListSessionsResponse,
        DeleteSessionResponse,
        ForkSessionResponse,
        ResumeSessionResponse,
        CloseSessionResponse,
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
    root: Union[AgentResponseMessage, AgentErrorMessage]
    """
    A JSON-RPC response object.
    """
