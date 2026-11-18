# Generated from schema/schema.json. Do not edit by hand.
# Schema ref: refs/tags/schema-v1.35.0

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from acp._deserialize import coerce_protocol_version, skip_invalid_items, use_default_on_error
from acp._schema_base import BaseModel, ElicitationContent
from pydantic import (
    AnyUrl,
    ConfigDict,
    Field,
    RootModel,
    StrictInt,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("mime_type", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("mime_type", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class Diff(BaseModel):
    path: str
    """
    The absolute file path being modified.
    """
    old_text: Annotated[Optional[str], Field(alias="oldText")] = None
    """
    The original content (omitted or `null` for new files).
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("old_text", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionReference(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The referenced session.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class ToolCallLocation(BaseModel):
    path: str
    """
    The absolute file path being accessed or modified.
    """
    line: Annotated[Optional[int], Field(ge=0)] = None
    """
    Optional 1-based line number within the file.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("line", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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
    format: Optional[Literal["email", "uri", "date", "date-time"]] = None
    """
    String format.

    Optional. Omitted and `null` are equivalent and mean there is no format constraint.
    """
    default: Optional[str] = None
    """
    Default value.

    Optional. Omitted and `null` are equivalent and mean no default value is provided.
    """
    enum: Optional[List[str]] = None
    """
    Enum values for untitled single-select enums.
    Optional. Omitted and `null` are equivalent and mean no untitled single-select choices are
    declared by `enum`.
    """
    one_of: Annotated[Optional[List[EnumOption]], Field(alias="oneOf")] = None
    """
    Titled enum options for titled single-select enums.
    Optional. Omitted and `null` are equivalent and mean no titled single-select choices are
    declared by `oneOf`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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
    enum: List[str]
    """
    Allowed enum values.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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


class PromptCapabilities(BaseModel):
    image: Optional[bool] = False
    """
    Agent supports `image` content blocks.
    """
    audio: Optional[bool] = False
    """
    Agent supports `audio` content blocks.
    """
    embedded_context: Annotated[Optional[bool], Field(alias="embeddedContext")] = False
    """
    Agent supports embedded context in `session/prompt` requests.

    When enabled, the Client is allowed to include `resource` content blocks
    in prompt requests for pieces of context that are referenced in the message.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("audio", "embedded_context", "image", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class McpCapabilities(BaseModel):
    http: Optional[bool] = False
    """
    Agent supports `http` MCP servers.
    """
    sse: Optional[bool] = False
    """
    Agent supports `sse` MCP servers.
    """
    acp: Optional[bool] = False
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Agent supports `acp` MCP servers.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("acp", "http", "sse", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SessionListCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionDeleteCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionAdditionalDirectoriesCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionSystemPromptOverrideCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionForkCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionArchiveCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionStatusCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionOptionsCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionTitleCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionResumeCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionCloseCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class AuthStatusCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class LogoutCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class ProvidersCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


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
    Optional human-readable description of this authentication method.
    """
    args: Optional[List[str]] = None
    """
    Additional arguments to append to the configured agent invocation for terminal auth.
    """
    env: Optional[Dict[str, str]] = None
    """
    Additional environment variables to set on the configured agent invocation for terminal auth.
    These values override same-named variables in the base launch configuration.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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
    Optional human-readable description of this authentication method.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class Implementation(BaseModel):
    name: str
    """
    Intended for programmatic or logical use, but can be used as a display
    name fallback if title isn't present.
    """
    title: Optional[str] = None
    """
    Intended for UI and end-user contexts and optimized to be human-readable
    and easily understood.

    If not provided, the name SHOULD be used for display.
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class AuthenticateResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class ProviderCurrentConfig(BaseModel):
    api_type: Annotated[
        Union[Literal["anthropic"], Literal["openai"], Literal["azure"], Literal["vertex"], Literal["bedrock"], str],
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SetProviderResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class DisableProviderResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class LogoutResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class AuthAccount(BaseModel):
    label: str
    """
    Human-readable name for the account, suitable for display on its own,
    for example `Claude Max`, `OpenAI API key`, or `Vertex AI`.
    """
    description: Optional[str] = None
    """
    Short text shown with the label, such as the credential source, for example
    `API key helper`. MUST NOT contain credentials.

    Optional. Omitted and `null` are equivalent and mean no description is reported.
    """
    active: Optional[bool] = None
    """
    Whether the agent currently uses this account by default.

    Several accounts MAY be active, for example one per provider.

    Optional. Omitted and `null` are equivalent and mean the agent does not indicate whether this account is in effect.
    """
    name: Optional[str] = None
    """
    Display name of the user, such as a username or full name.

    Optional. Omitted and `null` are equivalent and mean no name is reported.
    """
    email: Optional[str] = None
    """
    Email address of the user.

    Optional. Omitted and `null` are equivalent and mean no email is reported.
    """
    organization: Optional[str] = None
    """
    Display name of the organization, team, or workspace the account is using.

    Optional. Omitted and `null` are equivalent and mean no organization is reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("active", "description", "email", "name", "organization", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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
    Additional workspace roots reported for this session. Each path MUST be absolute.

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
    archived: Optional[bool] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the session is archived.

    Optional. Omitted and `null` are equivalent and mean the archive state is not reported.
    Agents that advertise the `sessionCapabilities.archive` capability MUST include it on every listed session.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("archived", "title", "updated_at", mode="wrap")
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class ArchiveSessionResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class UnarchiveSessionResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SetSessionTitleResponse(BaseModel):
    title: str
    """
    The title the Agent stored.

    This differs from the requested title if the Agent normalized it. Clients SHOULD
    display this value rather than the one they sent.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class CloseSessionResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SetSessionModeResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("cached_read_tokens", "cached_write_tokens", "thought_tokens", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class PlanMarkdown(BaseModel):
    plan_id: Annotated[str, Field(alias="planId")]
    """
    The plan ID to update.
    """
    content: str
    """
    The complete plan as Markdown, not just the text that changed.
    """
    path: Optional[str] = None
    """
    The absolute path of the file that holds this plan.

    Omitted and `null` both mean the plan has no file. A supplied path MUST
    be absolute. `content` is the plan to display whether or not `path` is set.
    Clients can use `path` to open the file.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("path", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class PlanRemoved(BaseModel):
    plan_id: Annotated[str, Field(alias="planId")]
    """
    The plan ID to remove. Clients MUST ignore a removal for an unknown ID.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class CurrentModeUpdateBase(BaseModel):
    current_mode_id: Annotated[str, Field(alias="currentModeId")]
    """
    The ID of the current mode
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionInfoUpdateBase(BaseModel):
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class Cost(BaseModel):
    amount: float
    """
    Total cumulative cost for session.
    """
    currency: str
    """
    Unit of `amount`: an ISO 4217 currency code (e.g., "USD", "EUR") for money, or another
    unit such as "credits".
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class Notice(BaseModel):
    severity: Union[Literal["info"], Literal["warning"], Literal["error"], str]
    """
    Presentation severity hint.
    """
    title: Annotated[str, Field(min_length=1)]
    """
    Required non-empty plain-text title that can stand alone.
    """
    description: Optional[str] = None
    """
    Optional plain-text detail or guidance.

    Omitted and `null` are equivalent and mean no description was supplied.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    Metadata scoped to this notice.

    Omitted and `null` are equivalent and mean no metadata was supplied.
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class CursorForkCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class CursorAnchor7(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: str
    """
    Custom or future anchor type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class CursorMessageAnchor(BaseModel):
    message_id: Annotated[str, Field(alias="messageId")]
    """
    The ID of the anchored message.
    """
    position: Union[Literal["before"], Literal["after"], str]
    """
    On which side of the message the cursor is.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class CursorToolCallAnchor(BaseModel):
    tool_call_id: Annotated[str, Field(alias="toolCallId")]
    """
    The ID of the anchored tool call.
    """
    position: Union[Literal["before"], Literal["after"], str]
    """
    On which side of the tool call the cursor is.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class CursorCompactionAnchor(BaseModel):
    compaction_id: Annotated[str, Field(alias="compactionId")]
    """
    The ID of the anchored compaction.
    """
    position: Union[Literal["before"], Literal["after"], str]
    """
    On which side of the compaction the cursor is.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionCancelCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class StateUpdate21(BaseModel):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[Literal["end_turn"], Field(alias="stopReason")] = "end_turn"
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class StateUpdate22(BaseModel):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[Literal["max_tokens"], Field(alias="stopReason")] = "max_tokens"
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class StateUpdate23(BaseModel):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[Literal["max_turn_requests"], Field(alias="stopReason")] = "max_turn_requests"
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class StateUpdate24(BaseModel):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[Literal["refusal"], Field(alias="stopReason")] = "refusal"
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class StateUpdate25(BaseModel):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[Literal["cancelled"], Field(alias="stopReason")] = "cancelled"
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class StateUpdate27(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[str, Field(alias="stopReason")]
    """
    Unrecognized stop reason.
    """


class StateUpdate28(BaseModel):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[None, Field(alias="stopReason")] = None
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class StateUpdate29(BaseModel):
    state: Literal["idle"] = "idle"


class StateUpdate210(StateUpdate21, StateUpdate29):
    state: Literal["idle"] = "idle"


class StateUpdate211(StateUpdate22, StateUpdate29):
    state: Literal["idle"] = "idle"


class StateUpdate212(StateUpdate23, StateUpdate29):
    state: Literal["idle"] = "idle"


class StateUpdate213(StateUpdate24, StateUpdate29):
    state: Literal["idle"] = "idle"


class StateUpdate214(StateUpdate25, StateUpdate29):
    state: Literal["idle"] = "idle"


class StateUpdate216(StateUpdate27, StateUpdate29):
    state: Literal["idle"] = "idle"


class StateUpdate217(StateUpdate28, StateUpdate29):
    state: Literal["idle"] = "idle"


class StateUpdate5(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    state: str
    """
    Unrecognized state discriminator.
    """


class RunningStateUpdate(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """


class IdleStateUpdate1(BaseModel):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[Literal["end_turn"], Field(alias="stopReason")] = "end_turn"
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class IdleStateUpdate2(BaseModel):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[Literal["max_tokens"], Field(alias="stopReason")] = "max_tokens"
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class IdleStateUpdate3(BaseModel):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[Literal["max_turn_requests"], Field(alias="stopReason")] = "max_turn_requests"
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class IdleStateUpdate4(BaseModel):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[Literal["refusal"], Field(alias="stopReason")] = "refusal"
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class IdleStateUpdate5(BaseModel):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[Literal["cancelled"], Field(alias="stopReason")] = "cancelled"
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class IdleStateUpdate7(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[str, Field(alias="stopReason")]
    """
    Unrecognized stop reason.
    """


class IdleStateUpdate8(BaseModel):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[None, Field(alias="stopReason")] = None
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class RequiresActionStateUpdate(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """


class UnknownStateUpdate(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class AuthStatusUpdateNotification(BaseModel):
    state: Union[Literal["authenticated"], Literal["unauthenticated"], Literal["not_required"], Literal["unknown"], str]
    """
    Whether the agent can start sessions without the user signing in first.

    Clients use this field to decide whether sign-in is needed. Clients MUST treat
    values they do not recognize as `unknown`.
    """
    accounts: Optional[List[AuthAccount]] = None
    """
    The accounts the agent has configured, for display.

    Optional. Omitted and empty are equivalent and mean no accounts are reported.
    Agents that hold credentials for several providers or accounts SHOULD list each
    of them and mark the ones in effect with `active`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("accounts", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("read_text_file", "write_text_file", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class CompactionCapabilities(BaseModel):
    pass


class BooleanConfigOptionCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class NoticeCapabilities(BaseModel):
    pass


class PromptSuggestionsImageCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class PromptSuggestionsAudioCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class PromptSuggestionsEmbeddedContextCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionCursorsCapabilities(BaseModel):
    pass


class SubagentCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.
    """


class PlanCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class AuthCapabilities(BaseModel):
    terminal: Optional[bool] = False
    """
    Whether the client supports `terminal` authentication methods.

    The client SHOULD set this to `true` only when it can reproduce the
    configured agent invocation in an interactive terminal. When `true`, the
    agent MAY include `terminal` entries in its authentication methods.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("terminal", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ElicitationFormCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class ElicitationUrlCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class ListProvidersRequest(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class LogoutRequest(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class AuthStatusRequest(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class InitialSessionConfigOption1(BaseModel):
    config_id: Annotated[str, Field(alias="configId")]
    """
    The ID of the configuration option to set.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """
    type: Literal["boolean"] = "boolean"
    value: bool
    """
    The boolean value.
    """


class InitialSessionConfigOption2(BaseModel):
    config_id: Annotated[str, Field(alias="configId")]
    """
    The ID of the configuration option to set.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """
    value: str
    """
    The value ID.
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class ArchiveSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to archive.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class UnarchiveSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to unarchive.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionStatusRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to check.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SessionOptionsRequest(BaseModel):
    cwd: str
    """
    The working directory to resolve options for. Must be an absolute path.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = None
    """
    Additional workspace roots to resolve options for. Each path MUST be absolute.

    Optional. Omitted and `null` are equivalent to an empty list.
    """
    session_id: Annotated[Optional[str], Field(alias="sessionId")] = None
    """
    The session to resolve options for.

    Optional. Omitted and `null` are equivalent and mean a new session. When present,
    the Agent returns what `session/load` or `session/resume` with the same `cwd` and
    `additionalDirectories` would return, with the same `cwd` rules. A session the
    Agent does not know gets `-32002`.
    Clients MUST only send this when the Agent supports `session/load` (`loadSession`)
    or `session/resume` (`sessionCapabilities.resume`).
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class SetSessionTitleRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to retitle.
    """
    title: str
    """
    The new title for the session.

    Required and non-null. Omission and explicit `null` are both invalid. The Agent MAY
    normalize the title before storing it, for example by trimming whitespace or
    truncating it. The Agent rejects a title that fails its validation, including one
    that is empty after normalization, with `-32602`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """
    type: Literal["boolean"] = "boolean"
    value: bool
    """
    The boolean value.
    """


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """
    value: str
    """
    The value ID.
    """


class WriteTextFileResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class DeniedOutcome(BaseModel):
    outcome: Literal["cancelled"] = "cancelled"


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("exit_code", "signal", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ReleaseTerminalResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("exit_code", "signal", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class KillTerminalResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class DeclineElicitationResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """
    action: Literal["decline"] = "decline"


class CancelElicitationResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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
    Allowed wire representations for `ElicitationContentValue`.
    """


class ElicitationAcceptAction(BaseModel):
    content: Optional[ElicitationContent] = None
    """
    The user-provided content, if any, as an object matching the requested schema.
    """


class McpResultResponse(BaseModel):
    result: Any
    """
    Required, even if JSON null.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    Optional ACP carrier metadata.
    """


class McpError(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    code: StrictInt
    """
    Inner MCP error code. Never an ACP error code.
    """
    message: str
    """
    Inner MCP error message.
    """
    data: Optional[Any] = None
    """
    Optional error data. Explicit null is retained.
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class MessageMcpNotification(BaseModel):
    server_id: Annotated[str, Field(alias="serverId")]
    """
    The declared ACP MCP server handling the associated request.
    """
    request_id: Annotated[str, Field(alias="requestId")]
    """
    The identifier of the active inner MCP request.
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class ProxySuccessorRequest(BaseModel):
    method: str
    """
    Method name of the wrapped request.
    """
    params: Optional[Any] = None
    """
    Params of the wrapped request, forwarded unchanged.

    Optional. Omitted and `null` are equivalent and mean the wrapped request has no params.
    When present, it MUST be a JSON object or array, as JSON-RPC requires.
    The wrapped request's own `_meta` stays inside these params.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    Metadata for the `proxy/successor` envelope itself, separate from the wrapped
    message's `_meta`. It is hop-local: the conductor does not copy it into the message
    it unwraps.

    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class ProxySuccessorNotification(BaseModel):
    method: str
    """
    Method name of the wrapped notification.
    """
    params: Optional[Any] = None
    """
    Params of the wrapped notification, forwarded unchanged.

    Optional. Omitted and `null` are equivalent and mean the wrapped notification has no params.
    When present, it MUST be a JSON object or array, as JSON-RPC requires.
    The wrapped notification's own `_meta` stays inside these params.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    Metadata for the `proxy/successor` envelope itself, separate from the wrapped
    message's `_meta`. It is hop-local: the conductor does not copy it into the message
    it unwraps.

    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class FileEditToolCallContent(Diff):
    type: Literal["diff"] = "diff"


class TerminalToolCallContent(Terminal):
    type: Literal["terminal"] = "terminal"


class ToolCallContentSession(SessionReference):
    type: Literal["session"] = "session"


class Annotations(BaseModel):
    audience: Optional[List[Literal["assistant", "user"]]] = None
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("annotations", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("annotations", "uri", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("annotations", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("annotations", "description", "mime_type", "size", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("annotations", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class PermissionOption(BaseModel):
    option_id: Annotated[str, Field(alias="optionId")]
    """
    Unique identifier for this permission option.
    """
    name: str
    """
    Human-readable label to display to the user.
    """
    kind: Literal["allow_once", "allow_always", "reject_once", "reject_always"]
    """
    Hint about the nature of this permission option.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("description", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("default", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class MessageMcpRequest(BaseModel):
    server_id: Annotated[str, Field(alias="serverId")]
    """
    The declared ACP MCP server receiving this request.
    """
    request_id: Annotated[str, Field(alias="requestId")]
    """
    The caller-generated identifier for the inner MCP request.
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
    session_id: Annotated[Optional[str], Field(alias="sessionId")] = None
    """
    The session whose work caused this request. It is the `sessionId` the
    Agent uses on `session/update` for that work. For a child session's work,
    that is the child's ID.

    Optional. Omitted and `null` are equivalent and mean the Agent supplied
    no session. The Agent SHOULD include it when it knows the session, and
    MUST NOT send another session's ID when it does not. A provider that
    needs the session to run the request MAY reject a request without it.
    """
    tool_call_id: Annotated[Optional[str], Field(alias="toolCallId")] = None
    """
    The tool call in that session that this request is part of. It is the
    `toolCallId` the Agent uses to report the call in `session/update`. The
    Agent SHOULD report the tool call before it sends the request.

    Optional. Omitted and `null` are equivalent and mean the Agent supplied
    no tool call. The Agent MUST NOT send it without `sessionId`. More than
    one request can name the same tool call.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("tool_call_id", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SessionSystemPromptModeCapabilities(BaseModel):
    override: Optional[SessionSystemPromptOverrideCapabilities] = None
    """
    Whether the agent supports `override` mode.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports replacing its built-in default
    instructions with the client-provided system prompt.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("override", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class AgentAuthCapabilities(BaseModel):
    status: Optional[AuthStatusCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the agent supports the `auth/status` method.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports the `auth/status` method.
    """
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("logout", "status", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class TerminalAuthMethod(AuthMethodTerminal):
    type: Literal["terminal"] = "terminal"


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
    If true, clients MUST NOT call `providers/disable` for this provider ID.
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("supported", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class AuthStatusResponse(BaseModel):
    state: Union[Literal["authenticated"], Literal["unauthenticated"], Literal["not_required"], Literal["unknown"], str]
    """
    Whether the agent can start sessions without the user signing in first.

    Clients use this field to decide whether sign-in is needed. Clients MUST treat
    values they do not recognize as `unknown`.
    """
    accounts: Optional[List[AuthAccount]] = None
    """
    The accounts the agent has configured, for display.

    Optional. Omitted and empty are equivalent and mean no accounts are reported.
    Agents that hold credentials for several providers or accounts SHOULD list each
    of them and mark the ones in effect with `active`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("accounts", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("available_modes", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


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
            Literal["mode"], Literal["model"], Literal["model_config"], Literal["speed"], Literal["thought_level"], str
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """
    type: Literal["boolean"] = "boolean"


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("next_cursor", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("sessions", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class SessionStatusResponse(BaseModel):
    status: Union[Literal["open"], Literal["saved"], str]
    """
    Whether the Agent has the session in memory.

    Required. Clients MUST treat a value they do not recognize as `saved`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class PromptResponse(BaseModel):
    stop_reason: Annotated[
        Literal["end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"], Field(alias="stopReason")
    ]
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("usage", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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
    This MUST be an integer as defined in the JSON-RPC specification.
    """
    message: str
    """
    A string providing a short description of the error.
    The message SHOULD be limited to a concise single sentence.
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


class AgentPlanRemovedUpdate(PlanRemoved):
    session_update: Annotated[Literal["plan_removed"], Field(alias="sessionUpdate")] = "plan_removed"


class CurrentModeUpdate(CurrentModeUpdateBase):
    session_update: Annotated[Literal["current_mode_update"], Field(alias="sessionUpdate")] = "current_mode_update"


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


class SessionUpdateNotice(Notice):
    session_update: Annotated[Literal["notice"], Field(alias="sessionUpdate")] = "notice"


class PlanEntry(BaseModel):
    content: str
    """
    Human-readable description of what this task aims to accomplish.
    """
    priority: Literal["high", "medium", "low"]
    """
    The relative importance of this task.
    Used to indicate which tasks are most critical to the overall goal.
    """
    status: Literal["pending", "in_progress", "completed"]
    """
    Current execution status of this task.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class Plan(BaseModel):
    entries: List[PlanEntry]
    """
    The list of tasks to be accomplished.

    The Agent MUST send all entries with their current status.
    The Client MUST replace the entire checklist with each update.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("entries", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class PlanUpdateMarkdown(PlanMarkdown):
    type: Literal["markdown"] = "markdown"


class PlanItems(BaseModel):
    plan_id: Annotated[str, Field(alias="planId")]
    """
    The plan ID to update.
    """
    entries: List[PlanEntry]
    """
    The list of tasks to be accomplished.

    The Agent MUST send all entries with their current status.
    The Client MUST replace the checklist with this `planId` on each update.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("entries", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class AvailableCommandInput(RootModel[UnstructuredCommandInput]):
    root: UnstructuredCommandInput
    """
    The input specification for a command.
    """


class CursorCapabilities(BaseModel):
    fork: Optional[CursorForkCapabilities] = None
    """
    Permits the Client to fork the session at this cursor with `session/fork`.

    Omitted and `null` are equivalent and mean unsupported. An object (including `{}`)
    means supported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("fork", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class CursorAnchor1(CursorMessageAnchor):
    type: Literal["user_message"] = "user_message"


class CursorAnchor2(CursorMessageAnchor):
    type: Literal["agent_message"] = "agent_message"


class CursorAnchor3(CursorMessageAnchor):
    type: Literal["agent_thought"] = "agent_thought"


class CursorAnchor4(CursorMessageAnchor):
    type: Literal["session_message"] = "session_message"


class CursorAnchor5(CursorToolCallAnchor):
    type: Literal["tool_call"] = "tool_call"


class CursorAnchor6(CursorCompactionAnchor):
    type: Literal["compaction"] = "compaction"


class CursorUpdate(BaseModel):
    cursor_id: Annotated[str, Field(alias="cursorId")]
    """
    The Agent-owned ID of this cursor, unique within the session.
    """
    capabilities: Optional[CursorCapabilities] = None
    """
    The methods that currently accept this cursor.

    Omission leaves the capabilities unchanged, `null` clears them, and an object replaces
    the whole previous value. A new cursor starts with none.
    """
    anchor: Optional[
        Union[CursorAnchor1, CursorAnchor2, CursorAnchor3, CursorAnchor4, CursorAnchor5, CursorAnchor6, CursorAnchor7]
    ] = None
    """
    Where the cursor is, relative to an entity the Client holds.

    Required on the first update for a cursor. The Agent MUST NOT change it afterwards.
    A later update can omit it or repeat the same anchor. Omitted and `null` are
    equivalent and leave the stored anchor unchanged.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    Extensible metadata patch for this cursor.
    """

    @field_validator("capabilities", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SubagentSessionCapabilities(BaseModel):
    cancel: Optional[SessionCancelCapabilities] = None
    """
    Permits the client to cancel this child's current work without ending
    the session. Omitted or `null` means unsupported. An object (including
    `{}`) means supported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.
    """

    @field_validator("cancel", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class StateUpdate1(RunningStateUpdate):
    state: Literal["running"] = "running"


class StateUpdate3(RequiresActionStateUpdate):
    state: Literal["requires_action"] = "requires_action"


class StateUpdate4(UnknownStateUpdate):
    state: Literal["unknown"] = "unknown"


class ErrorStopReason(BaseModel):
    error: Optional[Error] = None
    """
    The failure, as a JSON-RPC error object.

    Optional. Omitted or `null` both mean the agent is not reporting failure details.
    Agents SHOULD include it.
    """

    @field_validator("error", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class IdleStateUpdate6(ErrorStopReason):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[Literal["error"], Field(alias="stopReason")] = "error"
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class IdleStateUpdate(
    RootModel[
        Union[
            IdleStateUpdate1,
            IdleStateUpdate2,
            IdleStateUpdate3,
            IdleStateUpdate4,
            IdleStateUpdate5,
            IdleStateUpdate6,
            IdleStateUpdate7,
            IdleStateUpdate8,
        ]
    ]
):
    root: Union[
        IdleStateUpdate1,
        IdleStateUpdate2,
        IdleStateUpdate3,
        IdleStateUpdate4,
        IdleStateUpdate5,
        IdleStateUpdate6,
        IdleStateUpdate7,
        IdleStateUpdate8,
    ]
    """
    Foreground work is not in progress. The parent can give the child more work.

    An omitted or `null` `stopReason` means not reported.
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("boolean", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class PromptSuggestionsCapabilities(BaseModel):
    image: Optional[PromptSuggestionsImageCapabilities] = None
    """
    Whether prompt suggestions may include `image` content blocks.

    Optional. Omitted or `null` both mean the client does not advertise support.
    Supplying `{}` means the client can offer suggestions that include images.
    """
    audio: Optional[PromptSuggestionsAudioCapabilities] = None
    """
    Whether prompt suggestions may include `audio` content blocks.

    Optional. Omitted or `null` both mean the client does not advertise support.
    Supplying `{}` means the client can offer suggestions that include audio.
    """
    embedded_context: Annotated[
        Optional[PromptSuggestionsEmbeddedContextCapabilities], Field(alias="embeddedContext")
    ] = None
    """
    Whether prompt suggestions may include `resource` content blocks.

    Optional. Omitted or `null` both mean the client does not advertise support.
    Supplying `{}` means the client can offer suggestions that include
    embedded resources.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("audio", "embedded_context", "image", mode="wrap")
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("form", "url", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class HttpMcpServer(McpServerHttp):
    type: Literal["http"] = "http"


class SseMcpServer(McpServerSse):
    type: Literal["sse"] = "sse"


class AcpMcpServer(McpServerAcp):
    type: Literal["acp"] = "acp"


class ListSessionsRequest(BaseModel):
    cwd: Optional[str] = None
    """
    Filter sessions by working directory. Must be an absolute path.
    """
    cursor: Optional[str] = None
    """
    Opaque cursor token from a previous response's nextCursor field for cursor-based pagination
    """
    archived: Optional[Union[Literal["exclude"], Literal["include"], Literal["only"], str]] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Filter sessions by archive state.

    Optional. Omitted and `null` are equivalent and mean `exclude`. Only send this when
    the Agent advertises the `sessionCapabilities.archive` capability. The filter combines with `cwd` and
    applies before pagination. Clients keep the same value when following `nextCursor`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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
    outcome: Literal["selected"] = "selected"


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("exit_status", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class AcceptElicitationResponse(ElicitationAcceptAction):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    Optional. Omitted and `null` are equivalent and mean no metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """
    action: Literal["accept"] = "accept"


class McpErrorResponse(BaseModel):
    error: McpError
    """
    Required, non-null MCP error object.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    Optional ACP carrier metadata.
    """


class ClientNotification(BaseModel):
    method: str
    """
    The notification method name.
    """
    params: Optional[Union[CancelNotification, MessageMcpNotification, Any]] = None
    """
    Method-specific notification parameters.
    """


class ProxyResponse2(BaseModel):
    id: Optional[Union[int, str]]
    """
    The id of the request this response answers.
    """
    error: Error
    """
    Method-specific error data.
    """


class ProxyNotification(BaseModel):
    method: str
    """
    The notification method name.
    """
    params: Optional[ProxySuccessorNotification] = None
    """
    Method-specific notification parameters.
    """


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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


class SessionSystemPromptCapabilities(BaseModel):
    mode: Optional[SessionSystemPromptModeCapabilities] = None
    """
    Delivery modes the agent supports in addition to `append`.

    Optional. Omitted or `null` both mean the agent supports only `append`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("mode", mode="wrap")
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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


class AgentPlanUpdate(Plan):
    session_update: Annotated[Literal["plan"], Field(alias="sessionUpdate")] = "plan"


class SessionUpdateCursorUpdate(CursorUpdate):
    session_update: Annotated[Literal["cursor_update"], Field(alias="sessionUpdate")] = "cursor_update"


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("message_id", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class PlanUpdateItems(PlanItems):
    type: Literal["items"] = "items"


class PlanUpdate(BaseModel):
    plan: Annotated[Union[PlanUpdateItems, PlanUpdateMarkdown], Field(discriminator="type")]
    """
    Replaces the whole plan with this `planId`, even if its `type` changes.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("input", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class AvailableCommandsUpdateBase(BaseModel):
    available_commands: Annotated[List[AvailableCommand], Field(alias="availableCommands")]
    """
    Commands the agent can execute
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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
    ] = None
    """
    Complete replacement user-displayable summary content for the compaction.
    """
    error: Optional[str] = None
    """
    Human-readable error details for the compaction.
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
    content: Annotated[
        Union[
            TextContentBlock, ImageContentBlock, AudioContentBlock, ResourceContentBlock, EmbeddedResourceContentBlock
        ],
        Field(discriminator="type"),
    ]
    """
    One content block to append.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    Metadata scoped to this chunk. Omission and `null` both mean absent.
    """


class PromptSuggestion(BaseModel):
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
        Field(min_length=1),
    ]
    """
    The content of the suggested prompt.

    Required and non-null. Must contain at least one content block. This is
    the same type as the `prompt` field of `session/prompt`, and the Client
    can send it in that field unchanged.

    Agents MUST only use `text` and `resource_link` content blocks, and
    other content block types that both the Agent's prompt capabilities
    and the Client's `session.promptSuggestions` capability advertise.
    Agents MUST NOT send a prompt whose only content is empty or
    whitespace-only text.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class PromptSuggestionsUpdate(BaseModel):
    suggestions: List[PromptSuggestion]
    """
    The suggestions the Client can offer, ordered most likely first.

    Required and non-null. An empty list withdraws the suggestions the
    Client is offering.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("suggestions", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class StateUpdate26(ErrorStopReason):
    usage: Optional[Usage] = None
    """
    **UNSTABLE** Token usage for completed foreground work.

    Optional. Omitted or `null` means not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP for additional metadata.
    Implementations MUST NOT make assumptions about values at these keys.
    Optional. Omitted and `null` mean no metadata for this state snapshot.
    """
    stop_reason: Annotated[Literal["error"], Field(alias="stopReason")] = "error"
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted or `null` both mean the agent is not reporting a stop reason.

    See protocol docs: [Work state](https://agentclientprotocol.com/protocol/v1/draft/subagents#work-state)
    """


class StateUpdate215(StateUpdate26, StateUpdate29):
    state: Literal["idle"] = "idle"


class SubagentUpdate(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The opaque session ID identifying the child in all ACP messages.
    """
    title: Optional[str] = None
    """
    The parent's human-readable display title for this child. It need not be unique.

    Omitted means unchanged. `null` clears it. If unset, the Client chooses
    a fallback presentation.
    """
    description: Optional[str] = None
    """
    The parent's human-readable description of the child's role or purpose.

    Omitted means unchanged. `null` clears it. If unset, the Client chooses
    a fallback presentation. This is current display metadata, not the
    history of instructions sent to the child.
    """
    capabilities: Optional[SubagentSessionCapabilities] = None
    """
    Client-initiated session mutations permitted for this subagent session.

    Omitted means unchanged. `null` clears the capability set and disables
    child mutations. If never supplied, no session mutations are permitted.
    Read-only operations retain their normal protocol semantics and capability
    requirements. A concrete object replaces the whole capability set.
    """
    state: Optional[
        Union[
            StateUpdate1,
            StateUpdate3,
            StateUpdate4,
            StateUpdate5,
            StateUpdate210,
            StateUpdate211,
            StateUpdate212,
            StateUpdate213,
            StateUpdate214,
            StateUpdate215,
            StateUpdate216,
            StateUpdate217,
        ]
    ] = None
    """
    Current state snapshot for the child session.

    Omitted means unchanged. `null` clears the current activity without
    asserting idle or sending an `unknown` snapshot. A concrete state
    replaces the previous state object wholesale. If never supplied, the
    current activity is unset/unconfirmed.
    """
    cwd: Optional[str] = None
    """
    The child's working directory. Must be an absolute path.

    Optional and nullable. Omitted means unchanged. `null` clears the
    reported value. While unset, the child inherits its immediate parent's
    effective working directory. Clients use this value for the child's
    root set and as the default terminal working directory.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = None
    """
    Additional workspace roots for the child. Each path MUST be absolute.

    Optional and nullable. Omitted means unchanged. `null` clears the
    reported value. While unset, the child inherits its immediate parent's
    effective additional directories, independently of `cwd`. A concrete
    array replaces the whole list. `[]` reports no additional directories.
    Clients use these values for the child's root set.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    Omitted means unchanged. `null` removes the metadata.
    """

    @field_validator("capabilities", "description", "state", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SessionMessage(BaseModel):
    message_id: Annotated[str, Field(alias="messageId")]
    """
    Identifier of this message within the enclosing session's transcript.
    """
    sender_session_id: Annotated[Optional[str], Field(alias="senderSessionId")] = None
    """
    The sending session.

    Omitted leaves the stored value unchanged. `null` clears it. A value replaces it.
    """
    recipient_session_id: Annotated[Optional[str], Field(alias="recipientSessionId")] = None
    """
    The receiving session.

    Omitted leaves the stored value unchanged. `null` clears it. A value replaces it.
    """
    sender_message_id: Annotated[Optional[str], Field(alias="senderMessageId")] = None
    """
    On an incoming view, the `messageId` of the same message's outgoing view in the
    sender's transcript.

    Omitted leaves the stored value unchanged. `null` clears it. A value replaces it. It can arrive
    after the message starts streaming.
    """
    recipient_message_id: Annotated[Optional[str], Field(alias="recipientMessageId")] = None
    """
    On an outgoing view, the `messageId` of the same message's incoming view in the
    recipient's transcript.

    Omitted leaves the stored value unchanged. `null` clears it. A value replaces it. It can arrive
    after the message starts streaming.
    """
    content: Optional[
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
        ]
    ] = None
    """
    Omitted leaves content unchanged. `null` or `[]` clears it.
    A non-empty array replaces all content.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    Omitted leaves metadata unchanged. `null` removes it.

    Implementations MUST NOT make assumptions about values in `_meta`.
    """

    @field_validator(
        "recipient_message_id", "recipient_session_id", "sender_message_id", "sender_session_id", mode="wrap"
    )
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("content", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class SessionMessageChunk(BaseModel):
    message_id: Annotated[str, Field(alias="messageId")]
    """
    Identifier of this message within the enclosing session's transcript.
    """
    content: Annotated[
        Union[
            TextContentBlock, ImageContentBlock, AudioContentBlock, ResourceContentBlock, EmbeddedResourceContentBlock
        ],
        Field(discriminator="type"),
    ]
    """
    A single content block appended to the message.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    Optional and nullable chunk-scoped metadata. Omitted or `null` means none.

    Implementations MUST NOT make assumptions about values in `_meta`.
    """


class ClientSessionCapabilities(BaseModel):
    compaction: Optional[CompactionCapabilities] = None
    """
    Support for ID-addressed context compaction updates. Omitted or `null`
    means unsupported. `{}` advertises the complete compaction contract.
    """
    config_options: Annotated[Optional[SessionConfigOptionsCapabilities], Field(alias="configOptions")] = None
    """
    Config option capabilities supported by the client.

    Omitted or `null` both mean the client does not advertise support for any
    config option extensions.
    """
    notices: Optional[NoticeCapabilities] = None
    """
    Support for live user-facing `notice` session updates.

    Optional. Omitted or `null` both mean the client does not advertise support.
    Supplying `{}` means the client can present notices to the user.
    """
    prompt_suggestions: Annotated[Optional[PromptSuggestionsCapabilities], Field(alias="promptSuggestions")] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Support for `prompt_suggestions_update` session updates.

    Optional. Omitted or `null` both mean the client does not advertise support.
    Supplying `{}` means the client can offer prompt suggestions made of
    `text` and `resource_link` content blocks.
    """
    cursors: Optional[SessionCursorsCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Support for `cursor_update` session updates.

    Optional. Omitted or `null` both mean the client does not advertise support.
    Supplying `{}` means the client handles `cursor_update`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("compaction", "config_options", "cursors", "notices", "prompt_suggestions", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SystemPrompt(BaseModel):
    mode: Union[Literal["append"], Literal["override"], str]
    """
    How the Agent combines `content` with its own instructions.

    Required. Agents MUST return an error for a mode they do not support rather than
    treating it as `append` or `override`.
    """
    content: List[
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
    The system prompt content.

    Agents advertising the capability MUST accept `text` content blocks.
    Clients MUST NOT send other content block types until a future
    capability allows them, and Agents MUST return an error for content
    blocks they do not support.

    Clients MUST NOT send an empty array. Agents MAY reject one with
    `-32602` without creating, loading, or resuming the session.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class LoadSessionRequest(BaseModel):
    mcp_servers: Annotated[
        List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]], Field(alias="mcpServers")
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
    Additional workspace roots to activate for this session. Each path MUST be absolute.

    When omitted or empty, no additional roots are activated. When non-empty,
    this is the complete resulting additional-root list for the loaded
    session. It MAY differ from any previously used or reported list as long as
    the request `cwd` matches the session's `cwd`.
    """
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to load.
    """
    system_prompt: Annotated[Optional[SystemPrompt], Field(alias="systemPrompt")] = None
    """
    System-level instructions for the loaded session.

    Optional. Omitted and `null` are equivalent. The Agent MAY then keep a
    system prompt it stored with the session or use its default instructions.
    Clients MUST only send this when the Agent advertises
    `sessionCapabilities.systemPrompt`, and SHOULD send the same value they used
    when the session was created.
    """
    config_options: Annotated[
        Optional[List[Union[InitialSessionConfigOption1, InitialSessionConfigOption2]]], Field(alias="configOptions")
    ] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Initial values for the session's config options.

    Optional. Omitted and `null` are equivalent to an empty list. Clients MUST only send
    this when the Agent advertises `sessionCapabilities.options`.

    The Agent applies the entries in list order, as if the Client had called
    `session/set_config_option` for each entry after the session was set up. Malformed
    entries and duplicate `configId` values fail the request with `-32602`. The Agent MUST
    ignore an entry whose `configId` is unknown, whose `type` does not match the option,
    or whose value is not allowed when it is applied. The `configOptions` in the response contain the values the session actually
    has. The Agent SHOULD NOT send `config_option_update` notifications for these entries.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class ForkSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to fork.
    """
    cursor_id: Annotated[Optional[str], Field(alias="cursorId")] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    The session cursor to fork at. The new session's history is the source
    session's history before the cursor.

    Optional. Omitted and `null` are equivalent and fork the whole session.
    Clients MUST only send the ID of a cursor that the Agent announced on the
    source session and that currently advertises `capabilities.fork`. The Agent
    rejects any other ID with `-32602`. It does not fall back to forking the
    whole session.
    """
    cwd: str
    """
    The working directory for this session. Must be an absolute path.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = None
    """
    Additional workspace roots to activate for this session. Each path MUST be absolute.

    When omitted or empty, no additional roots are activated. When non-empty,
    this is the complete resulting additional-root list for the forked
    session.
    """
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]]], Field(alias="mcpServers")
    ] = None
    """
    List of MCP servers to connect to for this session.
    """
    system_prompt: Annotated[Optional[SystemPrompt], Field(alias="systemPrompt")] = None
    """
    System-level instructions for the forked session.

    Optional. Omitted and `null` are equivalent. The Agent MAY then keep a
    system prompt it stored with the source session or use its default
    instructions. Clients MUST only send this when the Agent advertises
    `sessionCapabilities.systemPrompt`, and SHOULD send the same value they used
    for the source session.
    """
    config_options: Annotated[
        Optional[List[Union[InitialSessionConfigOption1, InitialSessionConfigOption2]]], Field(alias="configOptions")
    ] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Initial values for the forked session's config options.

    Optional. Omitted and `null` are equivalent to an empty list. Clients MUST only send
    this when the Agent advertises `sessionCapabilities.options`.

    The Agent applies the entries in list order on top of the configuration the fork
    starts with, as if the Client had called `session/set_config_option` for each entry
    after the session was set up. Malformed entries and duplicate `configId` values fail
    the request with `-32602`. The Agent MUST ignore an entry whose `configId` is unknown,
    whose `type` does not match the option, or whose value is not allowed when it is
    applied. The `configOptions` in the response contain the values the session actually
    has. The Agent SHOULD NOT send `config_option_update` notifications for these entries,
    and SHOULD NOT send other `session/update` notifications caused by applying them
    before it responds.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


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
    Additional workspace roots to activate for this session. Each path MUST be absolute.

    When omitted or empty, no additional roots are activated. When non-empty,
    this is the complete resulting additional-root list for the resumed
    session. It MAY differ from any previously used or reported list as long as
    the request `cwd` matches the session's `cwd`.
    """
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]]], Field(alias="mcpServers")
    ] = None
    """
    List of MCP servers to connect to for this session.
    """
    system_prompt: Annotated[Optional[SystemPrompt], Field(alias="systemPrompt")] = None
    """
    System-level instructions for the resumed session.

    Optional. Omitted and `null` are equivalent. The Agent MAY then keep a
    system prompt it stored with the session or use its default instructions.
    Clients MUST only send this when the Agent advertises
    `sessionCapabilities.systemPrompt`, and SHOULD send the same value they used
    when the session was created.
    """
    config_options: Annotated[
        Optional[List[Union[InitialSessionConfigOption1, InitialSessionConfigOption2]]], Field(alias="configOptions")
    ] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Initial values for the session's config options.

    Optional. Omitted and `null` are equivalent to an empty list. Clients MUST only send
    this when the Agent advertises `sessionCapabilities.options`.

    The Agent applies the entries in list order, as if the Client had called
    `session/set_config_option` for each entry after the session was set up. Malformed
    entries and duplicate `configId` values fail the request with `-32602`. The Agent MUST
    ignore an entry whose `configId` is unknown, whose `type` does not match the option,
    or whose value is not allowed when it is applied. The `configOptions` in the response contain the values the session actually
    has. The Agent SHOULD NOT send `config_option_update` notifications for these entries.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


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

    As a baseline, the Agent MUST support `text` and `resource_link` content blocks,
    while other variants are optionally enabled via `PromptCapabilities`.

    The Client MUST adapt its interface according to `PromptCapabilities`.

    The client MAY include referenced pieces of context as either
    `resource` or `resource_link` content blocks.

    When available, `resource` is preferred
    as it avoids extra round-trips and allows the message to include
    pieces of context from sources the agent may not have access to.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class RequestPermissionResponse(BaseModel):
    outcome: Annotated[Union[DeniedOutcome, AllowedOutcome], Field(discriminator="outcome")]
    """
    The user's decision on the permission request.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


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
    Property definitions (MUST be primitive types).
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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
        Optional[SessionAdditionalDirectoriesCapabilities], Field(alias="additionalDirectories")
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
    system_prompt: Annotated[Optional[SessionSystemPromptCapabilities], Field(alias="systemPrompt")] = None
    """
    Whether the agent supports `systemPrompt` on session lifecycle requests.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports `systemPrompt` in `append` mode.
    """
    fork: Optional[SessionForkCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the agent supports `session/fork`.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports forking sessions.
    """
    archive: Optional[SessionArchiveCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the agent supports `session/archive` and `session/unarchive`.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports both methods, the `archived` filter on
    `session/list`, and `SessionInfo.archived`. Agents that advertise it MUST also advertise `sessionCapabilities.list`.
    """
    status: Optional[SessionStatusCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the agent supports `session/status`.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports checking session status.
    """
    options: Optional[SessionOptionsCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the agent supports `session/options`.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports `session/options` and accepts initial
    config option values on session setup requests.
    """
    title: Optional[SessionTitleCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the agent supports `session/set_title`.

    Agents report titles through `session/list` and `session_info_update` without this
    capability. It only advertises that Clients can set titles.

    Optional. Omitted or `null` both mean the agent does not advertise support.
    Supplying `{}` means the agent supports setting session titles.
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator(
        "additional_directories",
        "archive",
        "close",
        "delete",
        "fork",
        "list",
        "options",
        "resume",
        "status",
        "system_prompt",
        "title",
        mode="wrap",
    )
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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
            Literal["mode"], Literal["model"], Literal["model_config"], Literal["speed"], Literal["thought_level"], str
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """
    type: Literal["select"] = "select"


class LoadSessionResponse(BaseModel):
    modes: Optional[SessionModeState] = None
    """
    Initial mode state if supported by the Agent

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/v1/draft/session-modes)
    """
    config_options: Annotated[
        Optional[
            List[Annotated[Union[SessionConfigOptionSelect, SessionConfigOptionBoolean], Field(discriminator="type")]]
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("modes", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class ForkSessionResponse(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    Unique identifier for the newly created forked session.
    """
    modes: Optional[SessionModeState] = None
    """
    Initial mode state if supported by the Agent

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/v1/draft/session-modes)
    """
    config_options: Annotated[
        Optional[
            List[Annotated[Union[SessionConfigOptionSelect, SessionConfigOptionBoolean], Field(discriminator="type")]]
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("modes", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class SessionOptionsResponse(BaseModel):
    modes: Optional[SessionModeState] = None
    """
    The mode state the session would start with, if the Agent supports modes.

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/v1/draft/session-modes)
    """
    config_options: Annotated[
        Optional[
            List[Annotated[Union[SessionConfigOptionSelect, SessionConfigOptionBoolean], Field(discriminator="type")]]
        ],
        Field(alias="configOptions"),
    ] = None
    """
    The config options the session would start with, with the same rules as on
    `NewSessionResponse`.

    Optional. Omitted, `null`, or empty means no config options are advertised.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("modes", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class ResumeSessionResponse(BaseModel):
    modes: Optional[SessionModeState] = None
    """
    Initial mode state if supported by the Agent

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/v1/draft/session-modes)
    """
    config_options: Annotated[
        Optional[
            List[Annotated[Union[SessionConfigOptionSelect, SessionConfigOptionBoolean], Field(discriminator="type")]]
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("modes", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class SetSessionConfigOptionResponse(BaseModel):
    config_options: Annotated[
        List[Annotated[Union[SessionConfigOptionSelect, SessionConfigOptionBoolean], Field(discriminator="type")]],
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class UserMessageChunk(ContentChunk):
    session_update: Annotated[Literal["user_message_chunk"], Field(alias="sessionUpdate")] = "user_message_chunk"


class AgentMessageChunk(ContentChunk):
    session_update: Annotated[Literal["agent_message_chunk"], Field(alias="sessionUpdate")] = "agent_message_chunk"


class AgentThoughtChunk(ContentChunk):
    session_update: Annotated[Literal["agent_thought_chunk"], Field(alias="sessionUpdate")] = "agent_thought_chunk"


class AgentPlanContentUpdate(PlanUpdate):
    session_update: Annotated[Literal["plan_update"], Field(alias="sessionUpdate")] = "plan_update"


class AvailableCommandsUpdate(AvailableCommandsUpdateBase):
    session_update: Annotated[Literal["available_commands_update"], Field(alias="sessionUpdate")] = (
        "available_commands_update"
    )

    @field_validator("available_commands", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class SessionUpdateCompactionUpdate(CompactionUpdate):
    session_update: Annotated[Literal["compaction_update"], Field(alias="sessionUpdate")] = "compaction_update"


class SessionUpdateCompactionSummaryChunk(CompactionSummaryChunk):
    session_update: Annotated[Literal["compaction_summary_chunk"], Field(alias="sessionUpdate")] = (
        "compaction_summary_chunk"
    )


class SessionUpdatePromptSuggestionsUpdate(PromptSuggestionsUpdate):
    session_update: Annotated[Literal["prompt_suggestions_update"], Field(alias="sessionUpdate")] = (
        "prompt_suggestions_update"
    )


class SessionUpdateSubagentUpdate(SubagentUpdate):
    session_update: Annotated[Literal["subagent_update"], Field(alias="sessionUpdate")] = "subagent_update"


class SessionUpdateSessionMessage(SessionMessage):
    session_update: Annotated[Literal["session_message"], Field(alias="sessionUpdate")] = "session_message"


class SessionUpdateSessionMessageChunk(SessionMessageChunk):
    session_update: Annotated[Literal["session_message_chunk"], Field(alias="sessionUpdate")] = "session_message_chunk"


class ToolCall(BaseModel):
    tool_call_id: Annotated[str, Field(alias="toolCallId")]
    """
    Unique identifier for this tool call within the session.
    """
    title: str
    """
    Human-readable title describing what the tool is doing.
    """
    name: Optional[str] = None
    """
    Programmatic name of the tool being invoked.

    This field is optional. Omitting it or sending `null` both mean that no
    tool name is available.
    """
    kind: Optional[
        Literal["read", "edit", "delete", "move", "search", "execute", "think", "fetch", "switch_mode", "other"]
    ] = None
    """
    The category of tool being invoked.
    Helps clients choose appropriate icons and UI treatment.
    """
    status: Optional[Literal["pending", "in_progress", "completed", "failed"]] = None
    """
    Current execution status of the tool call.
    """
    content: Optional[
        List[
            Annotated[
                Union[ContentToolCallContent, FileEditToolCallContent, TerminalToolCallContent, ToolCallContentSession],
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("kind", "name", "raw_input", "raw_output", "status", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("content", "locations", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class ConfigOptionUpdateBase(BaseModel):
    config_options: Annotated[
        List[Annotated[Union[SessionConfigOptionSelect, SessionConfigOptionBoolean], Field(discriminator="type")]],
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class ClientCapabilities(BaseModel):
    fs: Annotated[Optional[FileSystemCapabilities], Field(validate_default=True)] = {
        "readTextFile": False,
        "writeTextFile": False,
    }
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
    subagents: Optional[SubagentCapabilities] = None
    """
    Whether the client understands exposed subagent sessions.

    Optional and nullable. Omitted or `null` both mean the client does not
    advertise support.
    Supplying `{}` means the client understands child associations, work-state
    snapshots, session-directed messages, and restricted-session semantics.

    See protocol docs: [Subagents](https://agentclientprotocol.com/protocol/v1/draft/subagents)
    """
    plan: Optional[PlanCapabilities] = None
    """
    Whether the Client supports `plan_update` and `plan_removed` session updates.

    Optional. Omitted or `null` both mean the Client does not advertise support.
    Supplying `{}` means the Client can receive both update types.
    """
    auth: Annotated[Optional[AuthCapabilities], Field(validate_default=True)] = {"terminal": False}
    """
    Authentication capabilities supported by the client.
    Determines which authentication method types the agent may include
    in its `InitializeResponse`.
    """
    elicitation: Optional[ElicitationCapabilities] = None
    """
    Elicitation capabilities supported by the client.
    Determines which elicitation modes the agent may use.

    Optional. Omitted or `null` both mean the client does not advertise
    elicitation support.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("auth", "elicitation", "fs", "plan", "session", "subagents", "terminal", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class NewSessionRequest(BaseModel):
    cwd: str
    """
    The working directory for this session. Must be an absolute path.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = None
    """
    Additional workspace roots for this session. Each path MUST be absolute.

    These expand the session's filesystem scope without changing `cwd`, which
    remains the base for relative paths. When omitted or empty, no
    additional roots are activated for the new session.
    """
    mcp_servers: Annotated[
        List[Union[HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio]], Field(alias="mcpServers")
    ]
    """
    List of MCP (Model Context Protocol) servers the agent should connect to.
    """
    system_prompt: Annotated[Optional[SystemPrompt], Field(alias="systemPrompt")] = None
    """
    System-level instructions for the new session.

    Optional. Omitted and `null` are equivalent and mean the Agent uses its
    own default instructions. Clients MUST only send this when the Agent
    advertises `sessionCapabilities.systemPrompt`.
    """
    config_options: Annotated[
        Optional[List[Union[InitialSessionConfigOption1, InitialSessionConfigOption2]]], Field(alias="configOptions")
    ] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Initial values for the session's config options.

    Optional. Omitted and `null` are equivalent to an empty list. Clients MUST only send
    this when the Agent advertises `sessionCapabilities.options`.

    The Agent applies the entries in list order, as if the Client had called
    `session/set_config_option` for each entry after the session was set up. Malformed
    entries and duplicate `configId` values fail the request with `-32602`. The Agent MUST
    ignore an entry whose `configId` is unknown, whose `type` does not match the option,
    or whose value is not allowed when it is applied. The `configOptions` in the response contain the values the session actually
    has. The Agent SHOULD NOT send `config_option_update` notifications for these entries,
    and SHOULD NOT send other `session/update` notifications caused by applying them
    before it responds.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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
        AcceptElicitationResponse,
        DeclineElicitationResponse,
        CancelElicitationResponse,
        OtherElicitationResponse,
        McpResultResponse,
        McpErrorResponse,
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


class ToolCallUpdate(BaseModel):
    tool_call_id: Annotated[str, Field(alias="toolCallId")]
    """
    The ID of the tool call being updated.
    """
    kind: Optional[
        Literal["read", "edit", "delete", "move", "search", "execute", "think", "fetch", "switch_mode", "other"]
    ] = None
    """
    Update the tool kind.
    """
    status: Optional[Literal["pending", "in_progress", "completed", "failed"]] = None
    """
    Update the execution status.
    """
    title: Optional[str] = None
    """
    Update the human-readable title.
    """
    name: Optional[str] = None
    """
    Update the programmatic name of the tool being invoked.

    This field is optional. Omitting it or sending `null` both mean that
    the existing name is left unchanged.
    """
    content: Optional[
        List[
            Annotated[
                Union[ContentToolCallContent, FileEditToolCallContent, TerminalToolCallContent, ToolCallContentSession],
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("kind", "name", "raw_input", "raw_output", "status", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("content", "locations", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """
    mode: Literal["form"] = "form"


class AgentCapabilities(BaseModel):
    load_session: Annotated[Optional[bool], Field(alias="loadSession")] = False
    """
    Whether the agent supports `session/load`.
    """
    prompt_capabilities: Annotated[
        Optional[PromptCapabilities], Field(alias="promptCapabilities", validate_default=True)
    ] = {"image": False, "audio": False, "embeddedContext": False}
    """
    Prompt capabilities supported by the agent.
    """
    mcp_capabilities: Annotated[Optional[McpCapabilities], Field(alias="mcpCapabilities", validate_default=True)] = {
        "http": False,
        "sse": False,
        "acp": False,
    }
    """
    MCP capabilities supported by the agent.
    """
    session_capabilities: Annotated[
        Optional[SessionCapabilities], Field(alias="sessionCapabilities", validate_default=True)
    ] = {}
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
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow clients and agents to attach additional
    metadata to their interactions. Implementations MUST NOT make assumptions about values at
    these keys.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator(
        "auth",
        "load_session",
        "mcp_capabilities",
        "prompt_capabilities",
        "providers",
        "session_capabilities",
        mode="wrap",
    )
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class NewSessionResponse(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    Unique identifier for the created session.

    Used in all subsequent requests for this conversation.
    """
    modes: Optional[SessionModeState] = None
    """
    Initial mode state if supported by the Agent

    See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/v1/draft/session-modes)
    """
    config_options: Annotated[
        Optional[
            List[Annotated[Union[SessionConfigOptionSelect, SessionConfigOptionBoolean], Field(discriminator="type")]]
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("modes", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class ToolCallStart(ToolCall):
    session_update: Annotated[Literal["tool_call"], Field(alias="sessionUpdate")] = "tool_call"


class ToolCallProgress(ToolCallUpdate):
    session_update: Annotated[Literal["tool_call_update"], Field(alias="sessionUpdate")] = "tool_call_update"


class ConfigOptionUpdate(ConfigOptionUpdateBase):
    session_update: Annotated[Literal["config_option_update"], Field(alias="sessionUpdate")] = "config_option_update"

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class InitializeRequest(BaseModel):
    protocol_version: Annotated[int, Field(alias="protocolVersion", ge=0, le=65535)]
    """
    The latest protocol version supported by the client.
    """
    client_capabilities: Annotated[
        Optional[ClientCapabilities], Field(alias="clientCapabilities", validate_default=True)
    ] = {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False, "auth": {"terminal": False}}
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("protocol_version", mode="before")
    @classmethod
    def coerce_protocol_version_validator(cls, v: Any, info: ValidationInfo) -> Any:
        return coerce_protocol_version(v, info)

    @field_validator("client_capabilities", "client_info", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class InitializeProxyRequest(InitializeRequest):
    pass


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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """


class InitializeResponse(BaseModel):
    protocol_version: Annotated[int, Field(alias="protocolVersion", ge=0, le=65535)]
    """
    The protocol version the client specified if supported by the agent,
    or the latest protocol version supported by the agent.

    The client SHOULD disconnect, if it doesn't support this version.
    """
    agent_capabilities: Annotated[
        Optional[AgentCapabilities], Field(alias="agentCapabilities", validate_default=True)
    ] = {
        "loadSession": False,
        "promptCapabilities": {"image": False, "audio": False, "embeddedContext": False},
        "mcpCapabilities": {"http": False, "sse": False, "acp": False},
        "sessionCapabilities": {},
        "auth": {},
    }
    """
    Capabilities supported by the agent.
    """
    auth_methods: Annotated[
        Optional[List[Union[TerminalAuthMethod, AuthMethodAgent]]], Field(alias="authMethods", validate_default=True)
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
    """

    @field_validator("agent_capabilities", "agent_info", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("auth_methods", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


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
            SessionUpdateNotice,
            SessionUpdateCompactionUpdate,
            SessionUpdateCompactionSummaryChunk,
            SessionUpdatePromptSuggestionsUpdate,
            SessionUpdateCursorUpdate,
            SessionUpdateSubagentUpdate,
            SessionUpdateSessionMessage,
            SessionUpdateSessionMessageChunk,
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

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v1/draft/extensibility)
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
            AuthStatusRequest,
            NewSessionRequest,
            LoadSessionRequest,
            ListSessionsRequest,
            DeleteSessionRequest,
            ForkSessionRequest,
            ArchiveSessionRequest,
            UnarchiveSessionRequest,
            SessionStatusRequest,
            SessionOptionsRequest,
            SetSessionTitleRequest,
            ResumeSessionRequest,
            CloseSessionRequest,
            SetSessionModeRequest,
            PromptRequest,
            SetSessionConfigOptionBooleanRequest,
            SetSessionConfigOptionSelectRequest,
            Any,
        ]
    ] = None
    """
    Method-specific request parameters.
    """


class ProxyRequest(BaseModel):
    id: Optional[Union[int, str]]
    """
    The request id used to correlate the matching response.
    """
    method: str
    """
    The method name to invoke.
    """
    params: Optional[Union[InitializeProxyRequest, ProxySuccessorRequest]] = None
    """
    Method-specific request parameters.
    """


class InitializeProxyResponse(InitializeResponse):
    pass


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
            MessageMcpRequest,
            CreateOtherSessionElicitationRequest,
            CreateOtherRequestElicitationRequest,
            CreateFormSessionElicitationRequest,
            CreateFormRequestElicitationRequest,
            CreateUrlSessionElicitationRequest,
            CreateUrlRequestElicitationRequest,
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
        AuthenticateResponse,
        ListProvidersResponse,
        SetProviderResponse,
        DisableProviderResponse,
        LogoutResponse,
        AuthStatusResponse,
        NewSessionResponse,
        LoadSessionResponse,
        ListSessionsResponse,
        DeleteSessionResponse,
        ForkSessionResponse,
        ArchiveSessionResponse,
        UnarchiveSessionResponse,
        SessionStatusResponse,
        SessionOptionsResponse,
        SetSessionTitleResponse,
        ResumeSessionResponse,
        CloseSessionResponse,
        SetSessionModeResponse,
        SetSessionConfigOptionResponse,
        PromptResponse,
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


class AgentNotification(BaseModel):
    method: str
    """
    The notification method name.
    """
    params: Optional[Union[SessionNotification, CompleteElicitationNotification, AuthStatusUpdateNotification, Any]] = (
        None
    )
    """
    Method-specific notification parameters.
    """


class ProxyResponse1(BaseModel):
    id: Optional[Union[int, str]]
    """
    The id of the request this response answers.
    """
    result: Union[InitializeProxyResponse, Any]
    """
    Method-specific response data.
    """


class ProxyResponse(RootModel[Union[ProxyResponse1, ProxyResponse2]]):
    root: Union[ProxyResponse1, ProxyResponse2]
    """
    A JSON-RPC response object.
    """


MessageMcpResponse = Annotated[Union[McpResultResponse, McpErrorResponse], Field(union_mode="left_to_right")]

PermissionOptionKind = Literal["allow_once", "allow_always", "reject_once", "reject_always"]
PlanEntryPriority = Literal["high", "medium", "low"]
PlanEntryStatus = Literal["pending", "in_progress", "completed"]
StopReason = Literal["end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"]
ToolCallStatus = Literal["pending", "in_progress", "completed", "failed"]
ToolKind = Literal[
    "read",
    "edit",
    "delete",
    "move",
    "search",
    "execute",
    "think",
    "fetch",
    "switch_mode",
    "other",
]

CreateOtherElicitationRequest = Union[
    CreateOtherSessionElicitationRequest,
    CreateOtherRequestElicitationRequest,
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
ElicitationMode = Union[
    ElicitationFormSessionMode,
    ElicitationFormRequestMode,
    ElicitationUrlSessionMode,
    ElicitationUrlRequestMode,
]

_AvailableCommandsUpdate = AvailableCommandsUpdateBase
_CurrentModeUpdate = CurrentModeUpdateBase
_ConfigOptionUpdate = ConfigOptionUpdateBase
_SessionInfoUpdate = SessionInfoUpdateBase
_UsageUpdate = UsageUpdateBase
_StringMultiSelectItems = StringMultiSelectItemsBase


class Jsonrpc(Enum):
    field_2_0 = "2.0"
