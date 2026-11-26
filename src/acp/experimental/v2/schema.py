# Generated from schema/v2/schema.json. Do not edit by hand.
# Schema ref: refs/tags/schema-v2.0.0-beta.4

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from acp._deserialize import coerce_protocol_version, empty_list_on_null, skip_invalid_items, use_default_on_error
from acp.experimental.v2._schema_base import (
    BaseModel,
    ElicitationContent,
    ErrorModel,
    IdleStateUpdateModel,
    InitialConfigRequestModel,
    MethodParamsModel,
    SessionConfigSelectModel,
)
from pydantic import (
    AnyUrl,
    AwareDatetime,
    ConfigDict,
    Field,
    RootModel,
    StrictBool,
    StrictInt,
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
    The contents of the resource. Its `mimeType`, when provided, describes the format.
    """
    uri: AnyUrl
    """
    URI associated with this resource or media payload.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type describing the encoded media payload.

    Omitted and `null` are equivalent and mean the MIME type is not specified.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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

    Omitted and `null` are equivalent and mean the MIME type is not specified.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("mime_type", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SkillReference(BaseModel):
    name: str
    """
    The `name` of a `skill` entry in the session's current command list.

    Required and non-null. The Agent MUST reject the prompt with `-32602`
    when it cannot resolve the name to a skill the user can invoke.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionReference(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The referenced session.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ToolCallLocation(BaseModel):
    path: str
    """
    The absolute file path being accessed or modified.
    """
    line: Annotated[Optional[int], Field(ge=1)] = None
    """
    Optional 1-based line number within the file. Omitted and `null` are equivalent and
    mean no specific line. `0` is invalid.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("line", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ToolCallCancelCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("terminal_id", "tool_call_id", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class CreateFormElicitationRequestBase(BaseModel):
    message: str
    """
    A plain-text message explaining to the user what the Agent needs and why.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    mode: Literal["form"] = "form"


class CreateUrlElicitationRequestBase(BaseModel):
    message: str
    """
    A plain-text message explaining to the user what the Agent needs and why.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    request_id: Annotated[Union[int, str], Field(alias="requestId")]
    """
    The pending Client-to-Agent request this elicitation is tied to.
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
    Human-readable title for this choice, in plain text. `const` holds the corresponding value.
    """
    description: Optional[str] = None
    """
    Plain-text description to help the user choose between options.

    Optional. Omitted and `null` are equivalent and mean no description is provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class StringPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional plain-text label for this field in the form.

    Optional. Omitted and `null` are equivalent and mean no title is provided.
    """
    description: Optional[str] = None
    """
    Plain-text help for the user filling out this field.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class NumberPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional plain-text label for this field in the form.

    Optional. Omitted and `null` are equivalent and mean no title is provided.
    """
    description: Optional[str] = None
    """
    Plain-text help for the user filling out this field.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class IntegerPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional plain-text label for this field in the form.

    Optional. Omitted and `null` are equivalent and mean no title is provided.
    """
    description: Optional[str] = None
    """
    Plain-text help for the user filling out this field.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("default", "description", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class BooleanPropertySchema(BaseModel):
    title: Optional[str] = None
    """
    Optional plain-text label for this field in the form.

    Optional. Omitted and `null` are equivalent and mean no title is provided.
    """
    description: Optional[str] = None
    """
    Plain-text help for the user filling out this field.

    Optional. Omitted and `null` are equivalent and mean no description is provided.
    """
    default: Optional[bool] = None
    """
    Default value.

    Optional. Omitted and `null` are equivalent and mean no default value is provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class TitledMultiSelectItems(BaseModel):
    any_of: Annotated[List[EnumOption], Field(alias="anyOf", min_length=1)]
    """
    Titled enum options. Must contain at least one option.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    URL-based elicitation mode where the Client directs the user to a URL.
    """


class ProtocolVersion(RootModel[int]):
    root: Annotated[int, Field(ge=0, le=65535)]
    """
    Protocol version identifier.

    This version is only bumped for breaking changes.
    Non-breaking changes should be introduced via capabilities.
    """


class Implementation(BaseModel):
    name: str
    """
    Name intended for programmatic or logical use. It can also serve as the
    plain-text display name when `title` is absent.
    """
    title: Optional[str] = None
    """
    Optional human-readable title chosen to make the implementation easy to
    recognize in the user interface. The title is plain text.

    Omission and `null` are equivalent. When displaying an implementation without a title,
    the receiver SHOULD use `name`.
    """
    version: str
    """
    Version of the implementation. Can be displayed to the user or used
    for debugging or metrics purposes. (e.g. "1.0.0").
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class PromptImageCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class PromptAudioCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class PromptEmbeddedContextCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class PromptSkillInlineCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class McpStdioCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class McpHttpCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class McpAcpCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionDeleteCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionAdditionalDirectoriesCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionSystemPromptOverrideCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionForkCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionArchiveCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionStatusCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionOptionsCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionTitleCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class AuthStatusCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ProvidersCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class AuthMethodTerminal(BaseModel):
    method_id: Annotated[str, Field(alias="methodId")]
    """
    Unique identifier for this authentication method.
    """
    name: str
    """
    Human-readable name of the authentication method, shown to users as plain text.
    """
    description: Optional[str] = None
    """
    Optional human-readable description of this authentication method.
    May contain Markdown.

    Omitted and `null` are equivalent and mean no description is provided.
    """
    args: Optional[List[str]] = []
    """
    Additional arguments to append to the configured Agent invocation for terminal auth.
    """
    env: Annotated[Optional[List[EnvVariable]], Field(validate_default=True)] = []
    """
    Additional environment variables to set on the configured Agent invocation for terminal auth.
    Names SHOULD be unique, and otherwise the last entry wins. These values override
    same-named variables in the base launch configuration.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("args", "env", mode="before")
    @classmethod
    def empty_list_on_null_validator(cls, v: Any, info: ValidationInfo) -> Any:
        return empty_list_on_null(v, info)


class AuthMethodAgent(BaseModel):
    method_id: Annotated[str, Field(alias="methodId")]
    """
    Unique identifier for this authentication method.
    """
    name: str
    """
    Human-readable name of the authentication method, shown to users as plain text.
    """
    description: Optional[str] = None
    """
    Optional human-readable description of this authentication method.
    May contain Markdown.

    Omitted and `null` are equivalent and mean no description is provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class LoginAuthResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SetProviderResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class DisableProviderResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class LogoutAuthResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class AuthAccount(BaseModel):
    label: str
    """
    Human-readable name for the account, suitable for display on its own as plain text,
    for example `Claude Max`, `OpenAI API key`, or `Vertex AI`.
    """
    description: Optional[str] = None
    """
    Short plain-text description shown with the label, such as the credential source
    `API key helper`. Agents MUST NOT put credentials in this field.

    Optional. Omitted and `null` are equivalent and mean no description is reported.
    """
    active: Optional[bool] = None
    """
    Whether the Agent currently uses this account by default.

    Several accounts MAY be active, for example one per provider.

    Optional. Omitted and `null` are equivalent and mean the Agent does not indicate whether this account is in effect.
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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("active", "description", "email", "name", "organization", mode="wrap")
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
    Human-readable label the Client can display for this option value, in plain text.
    """
    description: Optional[str] = None
    """
    Optional description for this option value.
    May contain Markdown.

    Omitted and `null` are equivalent and mean no description is provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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


class OtherAvailableCommand(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: str
    """
    Custom or future command type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """
    name: str
    """
    Plain-text command name. MUST NOT contain whitespace and MUST be unique within the command list.
    """
    description: str
    """
    Human-readable description of what the command does.
    May contain Markdown.
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
    Plain-text hint about the input the command expects. The Client can display it
    when the user hasn't provided input yet.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class Skill(BaseModel):
    name: str
    """
    Plain-text skill name. The Client uses it to refer to the skill in a `skill` content block.

    MUST NOT contain whitespace and MUST be unique within the command list,
    across commands and skills. It need not match the `name` in the skill's
    frontmatter.
    """
    description: str
    """
    Human-readable description of what the skill does, shown to users.
    May contain Markdown.
    """
    path: Optional[str] = None
    """
    The absolute path of the file that defines the skill, normally its
    `SKILL.md`, on the machine the Agent runs on.

    Omitted and `null` are equivalent and mean the skill has no such file,
    as with a skill that is built into the Agent or served by an MCP server.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("path", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SessionInfo(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    Unique identifier for the session
    """
    cwd: str
    """
    The working directory for this session. Must be an absolute path.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = []
    """
    Additional workspace roots reported for this session. Each path MUST be absolute.

    When present, this is the complete ordered additional-root list reported
    by the Agent. Omitted and empty values are equivalent: the response
    reports no additional roots.
    """
    title: Optional[str] = None
    """
    Human-readable title for the session, in plain text.

    Omitted and `null` are equivalent and mean no title is provided.
    """
    updated_at: Annotated[Optional[AwareDatetime], Field(alias="updatedAt")] = None
    """
    RFC 3339 timestamp of last activity.

    Omitted and `null` are equivalent and mean the last activity time is not reported.
    """
    archived: Optional[bool] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the session is archived.

    Optional. Omitted and `null` are equivalent and mean the archive state is not reported.
    Agents that advertise the `session.archive` capability MUST include it on every listed session.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("archived", "title", "updated_at", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)

    @field_validator("additional_directories", mode="before")
    @classmethod
    def empty_list_on_null_validator(cls, v: Any, info: ValidationInfo) -> Any:
        return empty_list_on_null(v, info)


class DeleteSessionResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ArchiveSessionResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class UnarchiveSessionResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


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


class OtherBackgroundWork(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    type: str
    """
    Custom or future background work type.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class BackgroundTerminal(BaseModel):
    terminal_id: Annotated[str, Field(alias="terminalId")]
    """
    The ID of the listed terminal.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class BackgroundToolCall(BaseModel):
    tool_call_id: Annotated[str, Field(alias="toolCallId")]
    """
    The ID of the listed tool call.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class BackgroundSubagent(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the listed session.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class UnsupportedProtocolVersionErrorData(BaseModel):
    requested: Annotated[int, Field(ge=0, le=65535)]
    """
    The protocol version the Client requested.
    """
    supported: List[ProtocolVersion]
    """
    The protocol versions the Agent supports.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionNotFoundErrorData(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID from the request.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class Usage(BaseModel):
    total_tokens: Annotated[int, Field(alias="totalTokens", ge=0)]
    """
    Sum of all token types for the completed work.
    """
    input_tokens: Annotated[int, Field(alias="inputTokens", ge=0)]
    """
    Input tokens for the completed work.
    """
    output_tokens: Annotated[int, Field(alias="outputTokens", ge=0)]
    """
    Output tokens for the completed work.
    """
    thought_tokens: Annotated[Optional[int], Field(alias="thoughtTokens", ge=0)] = None
    """
    Thought/reasoning tokens for the completed work.

    Omitted and `null` are equivalent and mean the count is not reported.
    """
    cached_read_tokens: Annotated[Optional[int], Field(alias="cachedReadTokens", ge=0)] = None
    """
    Cache read tokens for the completed work.

    Omitted and `null` are equivalent and mean the count is not reported.
    """
    cached_write_tokens: Annotated[Optional[int], Field(alias="cachedWriteTokens", ge=0)] = None
    """
    Cache write tokens for the completed work.

    Omitted and `null` are equivalent and mean the count is not reported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("cached_read_tokens", "cached_write_tokens", "thought_tokens", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SetSessionTitleResponse(BaseModel):
    title: str
    """
    The plain-text title the Agent stored.

    This differs from the requested title if the Agent normalized it. Clients SHOULD
    display this value rather than the one they sent.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CloseSessionResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CancelToolCallResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CancelTerminalResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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


class TerminalOutput(BaseModel):
    data: Annotated[str, Field(json_schema_extra={"contentEncoding": "base64"})]
    """
    Base64-encoded replacement terminal output bytes.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class TerminalExitStatus(BaseModel):
    exit_code: Annotated[Optional[int], Field(alias="exitCode")] = None
    """
    Process exit code as the platform reports it, when known. Negative values are
    allowed, for example signed Windows exit codes. Omitted and `null` are equivalent.
    """
    signal: Optional[str] = None
    """
    Signal that terminated the process, when known.

    Agents SHOULD use the conventional platform signal name. POSIX examples
    include `SIGTERM`, `SIGKILL`, and `SIGINT`. Other platforms may use a
    platform-specific name. Omitted and `null` are equivalent.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("exit_code", "signal", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class TerminalSize(BaseModel):
    cols: Annotated[int, Field(ge=1, le=65535)]
    """
    Number of columns. At least 1.
    """
    rows: Annotated[int, Field(ge=1, le=65535)]
    """
    Number of rows. At least 1.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class TerminalCancelCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.
    This metadata is scoped to the chunk and does not patch stored metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionInfoUpdateBase(BaseModel):
    title: Optional[str] = None
    """
    Human-readable title for the session, in plain text. Set to `null` to clear.
    """
    updated_at: Annotated[Optional[AwareDatetime], Field(alias="updatedAt")] = None
    """
    RFC 3339 timestamp of last activity. Set to null to clear.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted leaves the stored metadata unchanged. `null`
    clears it. An object patches the stored metadata by top-level key: keys it omits are
    unchanged, a key set to `null` is removed, and any other value replaces that key's
    stored value without merging nested objects. Beyond this patch rule, implementations
    MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Metadata Patches](https://agentclientprotocol.com/protocol/v2/draft/extensibility#metadata-patches)
    """

    @field_validator("title", "updated_at", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    Cumulative session cost.

    Omitted leaves the last reported cost unchanged. `null` clears it. A value replaces it.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted leaves the stored metadata unchanged. `null`
    clears it. An object patches the stored metadata by top-level key: keys it omits are
    unchanged, a key set to `null` is removed, and any other value replaces that key's
    stored value without merging nested objects. Beyond this patch rule, implementations
    MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Metadata Patches](https://agentclientprotocol.com/protocol/v2/draft/extensibility#metadata-patches)
    """

    @field_validator("cost", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ReplayComplete(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CursorForkCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class OtherCursorAnchor(BaseModel):
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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionCancelCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CompleteElicitationNotification(BaseModel):
    elicitation_id: Annotated[str, Field(alias="elicitationId")]
    """
    The ID of the elicitation that completed.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class AuthStatusUpdateNotification(BaseModel):
    state: Union[Literal["authenticated"], Literal["unauthenticated"], Literal["not_required"], Literal["unknown"], str]
    """
    Whether the Agent can start sessions without the user signing in first.

    Clients use this field to decide whether sign-in is needed. Clients MUST treat
    values they do not recognize as `unknown`.
    """
    accounts: Optional[List[AuthAccount]] = None
    """
    The accounts the Agent has configured, for display.

    Optional. Omitted and empty are equivalent and mean no accounts are reported.
    Agents that hold credentials for several providers or accounts SHOULD list each
    of them and mark the ones in effect with `active`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("accounts", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class TerminalAuthCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ElicitationFormCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ElicitationUrlCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class PromptSuggestionsImageCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class PromptSuggestionsAudioCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class PromptSuggestionsEmbeddedContextCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CommandsSkillCapabilities(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ListProvidersRequest(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class DisableProviderRequest(BaseModel):
    provider_id: Annotated[str, Field(alias="providerId")]
    """
    Provider ID to disable.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class LogoutAuthRequest(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class AuthStatusRequest(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class McpServerHttp(BaseModel):
    name: str
    """
    Human-readable name identifying this MCP server. MUST be unique within `mcpServers`.

    Agents reject a request with a duplicate `name` with `-32602`.
    """
    url: AnyUrl
    """
    URL to the MCP server.
    """
    headers: Annotated[Optional[List[HttpHeader]], Field(validate_default=True)] = []
    """
    HTTP headers to set when making requests to the MCP server. Headers the MCP transport
    defines are owned by the transport, and Agents use the transport's values for them.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("headers", mode="before")
    @classmethod
    def empty_list_on_null_validator(cls, v: Any, info: ValidationInfo) -> Any:
        return empty_list_on_null(v, info)


class McpServerAcp(BaseModel):
    name: str
    """
    Human-readable name identifying this MCP server. MUST be unique within `mcpServers`.

    Agents reject a request with a duplicate `name` with `-32602`.
    """
    server_id: Annotated[str, Field(alias="serverId")]
    """
    Unique identifier for this MCP server, generated by the component providing it.

    Providers MUST NOT reuse an ID for multiple ACP-transport MCP servers that are visible
    on the same ACP connection.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class McpServerStdio(BaseModel):
    name: str
    """
    Human-readable name identifying this MCP server. MUST be unique within `mcpServers`.

    Agents reject a request with a duplicate `name` with `-32602`.
    """
    command: str
    """
    Path to the MCP server executable. Must be an absolute path.

    Agents reject a request with a relative `command` with `-32602`.
    """
    args: Optional[List[str]] = []
    """
    Command-line arguments to pass to the MCP server.
    """
    env: Annotated[Optional[List[EnvVariable]], Field(validate_default=True)] = []
    """
    Environment variables to set when launching the MCP server. Names SHOULD be unique.
    Agents MAY reject duplicates with `-32602`, and otherwise the last entry wins.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("args", "env", mode="before")
    @classmethod
    def empty_list_on_null_validator(cls, v: Any, info: ValidationInfo) -> Any:
        return empty_list_on_null(v, info)


class InitialIdConfigOption(BaseModel):
    config_id: Annotated[str, Field(alias="configId")]
    """
    The ID of the configuration option to set.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    type: Literal["id"] = "id"
    value: str
    """
    The value ID.
    """


class InitialBooleanConfigOption(BaseModel):
    config_id: Annotated[str, Field(alias="configId")]
    """
    The ID of the configuration option to set.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    type: Literal["boolean"] = "boolean"
    value: StrictBool
    """
    The boolean value.
    """


class OtherInitialConfigOption(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    config_id: Annotated[str, Field(alias="configId")]
    """
    The ID of the configuration option to set.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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


class DeleteSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to delete.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ArchiveSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to archive.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class UnarchiveSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to unarchive.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionStatusRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to check.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionOptionsRequest(BaseModel):
    cwd: str
    """
    The working directory to resolve options for. Must be an absolute path.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = []
    """
    Additional workspace roots to resolve options for. Each path MUST be absolute.

    Optional. Omitted and `null` are equivalent to an empty list.
    """
    session_id: Annotated[Optional[str], Field(alias="sessionId")] = None
    """
    The session to resolve options for.

    Optional. Omitted and `null` are equivalent and mean a new session. When present,
    the Agent returns what `session/resume` with the same `cwd` and
    `additionalDirectories` would return, with the same `cwd` rules. A session the
    Agent does not know gets `-32002`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("additional_directories", mode="before")
    @classmethod
    def empty_list_on_null_validator(cls, v: Any, info: ValidationInfo) -> Any:
        return empty_list_on_null(v, info)


class SetSessionTitleRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to retitle.
    """
    title: str
    """
    The new plain-text title for the session.

    Required and non-null. Omission and explicit `null` are both invalid. The Agent MAY
    normalize the title before storing it, for example by trimming whitespace or
    truncating it. The Agent rejects a title that fails its validation, including one
    that is empty after normalization, with `-32602`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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


class ReplayFromStart(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CloseSessionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to close.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CancelToolCallRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session that owns the tool call.
    """
    tool_call_id: Annotated[str, Field(alias="toolCallId")]
    """
    The ID of the tool call to cancel.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CancelTerminalRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session that owns the terminal.
    """
    terminal_id: Annotated[str, Field(alias="terminalId")]
    """
    The ID of the terminal to cancel.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    type: Literal["id"] = "id"
    value: str
    """
    The value ID.
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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    type: Literal["boolean"] = "boolean"
    value: StrictBool
    """
    The boolean value.
    """


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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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


class CancelledPermissionOutcomeBase(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SelectedPermissionOutcome(BaseModel):
    option_id: Annotated[str, Field(alias="optionId")]
    """
    The ID of the option the user selected from this request's `options`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class DeclineElicitationResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    action: Literal["decline"] = "decline"


class CancelElicitationResponse(BaseModel):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    action: Literal["cancel"] = "cancel"


class OtherElicitationResponse(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    action: str
    """
    Custom or future elicitation action.

    Values beginning with `_` are reserved for implementation-specific
    extensions. Unknown values that do not begin with `_` are reserved for
    future ACP variants.
    """


class ElicitationContentValue(RootModel[Union[str, int, float, bool, List[str], Any]]):
    root: Union[str, int, float, bool, List[str], Any]
    """
    Allowed wire representations for `ElicitationContentValue`.
    """


class ElicitationAcceptAction(BaseModel):
    content: Optional[ElicitationContent] = None
    """
    The user-provided content, if any, as an object matching the requested schema.

    Omitted and `null` are equivalent and mean no content is provided.
    """


class McpResultResponse(BaseModel):
    result: Any
    """
    Required, even if JSON null.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.
    This is ACP metadata, not the inner MCP message's `_meta`.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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


class CancelSessionNotification(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session whose foreground work to cancel.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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
    Optional inner MCP params. Null is equivalent to omission.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.
    This is ACP metadata, not the inner MCP message's `_meta`.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CancelRequestNotification(BaseModel):
    request_id: Annotated[Union[int, str], Field(alias="requestId")]
    """
    The ID of the request to cancel: a request that the sender of this notification issued.
    Receivers match it only against requests they received from that sender.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class CommandPermissionSubjectVariant(CommandPermissionSubject):
    type: Literal["command"] = "command"


class TerminalToolCallContent(Terminal):
    type: Literal["terminal"] = "terminal"


class SessionToolCallContent(SessionReference):
    type: Literal["session"] = "session"


class SkillContentBlock(SkillReference):
    type: Literal["skill"] = "skill"


class Annotations(BaseModel):
    audience: Optional[List[Union[Literal["assistant"], Literal["user"], str]]] = None
    """
    Intended recipients for this content, such as the user or assistant.

    Omitted and `null` are equivalent and mean no audience is specified.
    """
    last_modified: Annotated[Optional[str], Field(alias="lastModified")] = None
    """
    Timestamp indicating when the underlying resource was last modified.

    Should be an ISO 8601 formatted string (e.g., "2025-01-12T15:00:58Z"), as in MCP.

    Omitted and `null` are equivalent and mean the modification time is not reported.
    """
    priority: Annotated[Optional[float], Field(ge=0.0, le=1.0)] = None
    """
    Relative importance of this content when Clients choose what to surface, from 0 (least
    important) to 1 (most important).

    Omitted and `null` are equivalent and mean no priority is specified.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    Text payload carried by this content block. It may be plain text or Markdown.
    """
    annotations: Optional[Annotations] = None
    """
    Optional annotations that help Clients decide how to display or route this content.

    Omitted and `null` are equivalent and mean no annotations are provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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

    Omitted and `null` are equivalent and mean no URI is associated with the image.
    """
    annotations: Optional[Annotations] = None
    """
    Optional annotations that help Clients decide how to display or route this content.

    Omitted and `null` are equivalent and mean no annotations are provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    Optional annotations that help Clients decide how to display or route this content.

    Omitted and `null` are equivalent and mean no annotations are provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("annotations", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class Icon(BaseModel):
    src: AnyUrl
    """
    A standard URI pointing to an icon resource. May be an HTTP/HTTPS URL or a
    `data:` URI with Base64-encoded image data.

    Clients MUST accept only `https:` and `data:` URIs, MUST fetch without
    credentials, and MUST NOT follow redirects that change the scheme or origin.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    Optional MIME type override if the source MIME type is missing or generic.

    Omitted and `null` are equivalent and mean the source MIME type is used.
    """
    sizes: Optional[List[str]] = None
    """
    Optional array of strings that specify sizes at which the icon can be used.
    Each string SHOULD be in `WxH` format (e.g., `"48x48"`, `"96x96"`) or
    `"any"` for scalable formats like SVG.

    If not provided, the Client SHOULD assume that the icon can be used at any size.

    Omitted and `null` are equivalent and mean no sizes are specified.
    """
    theme: Optional[Union[Literal["light"], Literal["dark"], str]] = None
    """
    Optional theme this icon is designed for.

    Omitted and `null` are equivalent and mean the icon is not theme-specific.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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
    Human-readable name shown for the linked resource, in plain text.
    """
    uri: AnyUrl
    """
    URI associated with this resource or media payload.
    """
    title: Optional[str] = None
    """
    Optional plain-text display title for the resource, intended for end users.

    Omitted and `null` are equivalent and mean no title is provided.
    """
    description: Optional[str] = None
    """
    Optional human-readable description shown with the link. It describes the resource
    contents in plain text.

    Omitted and `null` are equivalent and mean no description is provided.
    """
    icons: Optional[List[Icon]] = None
    """
    Optional set of sized icons that the Client can display in a user interface.

    Omitted and `null` are equivalent and mean no icons are provided.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type describing the encoded media payload.

    Omitted and `null` are equivalent and mean the MIME type is not specified.
    """
    size: Annotated[Optional[int], Field(ge=0)] = None
    """
    Optional size of the linked resource in bytes, if known.

    Omitted and `null` are equivalent and mean the size is not known.
    """
    annotations: Optional[Annotations] = None
    """
    Optional annotations that help Clients decide how to display or route this content.

    Omitted and `null` are equivalent and mean no annotations are provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    Optional annotations that help Clients decide how to display or route this content.

    Omitted and `null` are equivalent and mean no annotations are provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    File content kind after the operation (for `delete`, of the deleted file).

    Omitted and `null` are equivalent and mean the content kind is unknown.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type of the file contents after the operation (for `delete`, of the deleted file).

    Omitted and `null` are equivalent and mean the MIME type is unknown.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    operation: Literal["add"] = "add"


class DeleteDiffChange(DiffPathChange):
    file_type: Annotated[
        Optional[Union[Literal["text"], Literal["binary"], Literal["directory"], Literal["symlink"], str]],
        Field(alias="fileType"),
    ] = None
    """
    File content kind after the operation (for `delete`, of the deleted file).

    Omitted and `null` are equivalent and mean the content kind is unknown.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type of the file contents after the operation (for `delete`, of the deleted file).

    Omitted and `null` are equivalent and mean the MIME type is unknown.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    operation: Literal["delete"] = "delete"


class ModifyDiffChange(DiffPathChange):
    file_type: Annotated[
        Optional[Union[Literal["text"], Literal["binary"], Literal["directory"], Literal["symlink"], str]],
        Field(alias="fileType"),
    ] = None
    """
    File content kind after the operation (for `delete`, of the deleted file).

    Omitted and `null` are equivalent and mean the content kind is unknown.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type of the file contents after the operation (for `delete`, of the deleted file).

    Omitted and `null` are equivalent and mean the MIME type is unknown.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    operation: Literal["modify"] = "modify"


class MoveDiffChange(DiffPathPairChange):
    file_type: Annotated[
        Optional[Union[Literal["text"], Literal["binary"], Literal["directory"], Literal["symlink"], str]],
        Field(alias="fileType"),
    ] = None
    """
    File content kind after the operation (for `delete`, of the deleted file).

    Omitted and `null` are equivalent and mean the content kind is unknown.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type of the file contents after the operation (for `delete`, of the deleted file).

    Omitted and `null` are equivalent and mean the MIME type is unknown.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    operation: Literal["move"] = "move"


class CopyDiffChange(DiffPathPairChange):
    file_type: Annotated[
        Optional[Union[Literal["text"], Literal["binary"], Literal["directory"], Literal["symlink"], str]],
        Field(alias="fileType"),
    ] = None
    """
    File content kind after the operation (for `delete`, of the deleted file).

    Omitted and `null` are equivalent and mean the content kind is unknown.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type of the file contents after the operation (for `delete`, of the deleted file).

    Omitted and `null` are equivalent and mean the MIME type is unknown.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    File content kind after the operation (for `delete`, of the deleted file).

    Omitted and `null` are equivalent and mean the content kind is unknown.
    """
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    """
    MIME type of the file contents after the operation (for `delete`, of the deleted file).

    Omitted and `null` are equivalent and mean the MIME type is unknown.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class Diff(BaseModel):
    changes: Annotated[
        List[Union[AddDiffChange, DeleteDiffChange, ModifyDiffChange, MoveDiffChange, CopyDiffChange, OtherDiffChange]],
        Field(min_length=1),
    ]
    """
    Structured file changes described by this diff. Agents MUST include at least one entry.

    Clients can use this field without parsing patch text to determine affected paths.
    """
    patch: Optional[DiffPatch] = None
    """
    Renderable patch text for some or all of the structured changes.

    Agents SHOULD provide patch text whenever feasible. Omitted and `null` are
    equivalent and mean no renderable patch text was provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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


class ToolCallCapabilities(BaseModel):
    cancel: Optional[ToolCallCancelCapabilities] = None
    """
    Permits the Client to cancel this tool call with `tool_call/cancel`.

    Omitted and `null` are equivalent and mean unsupported. An object (including `{}`)
    means supported. Agents SHOULD stop advertising it once the tool call reaches a final status.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("cancel", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class PermissionOption(BaseModel):
    option_id: Annotated[str, Field(alias="optionId")]
    """
    Identifier for this permission option. MUST be unique within the request.
    """
    name: str
    """
    Human-readable label for this choice, shown to the user as plain text.
    """
    kind: Union[Literal["allow_once"], Literal["allow_always"], Literal["reject_once"], Literal["reject_always"], str]
    """
    What selecting this option means. Clients can also use it to choose how to display the option.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    A plain-text message explaining to the user what the Agent needs and why.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    mode: Literal["url"] = "url"


class CreateUrlRequestElicitationRequest(CreateUrlRequestElicitationRequestBase, CreateUrlElicitationRequestBase):
    message: str
    """
    A plain-text message explaining to the user what the Agent needs and why.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    mode: Literal["url"] = "url"


class CreateOtherSessionElicitationRequest(ElicitationSessionScope):
    model_config = ConfigDict(
        extra="allow",
    )
    message: str
    """
    A plain-text message explaining to the user what the Agent needs and why.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    A plain-text message explaining to the user what the Agent needs and why.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    Optional plain-text label for this field in the form.

    Optional. Omitted and `null` are equivalent and mean no title is provided.
    """
    description: Optional[str] = None
    """
    Plain-text help for the user filling out this field.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    Optional inner MCP params. Null is equivalent to omission.
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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.
    This is ACP metadata, not the inner MCP message's `_meta`.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("tool_call_id", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class PromptSkillCapabilities(BaseModel):
    inline: Optional[PromptSkillInlineCapabilities] = None
    """
    Agent supports `skill` content blocks at any position in a prompt.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means a prompt may contain any number of `skill` content blocks, at any
    position.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("inline", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class McpCapabilities(BaseModel):
    stdio: Optional[McpStdioCapabilities] = None
    """
    Agent supports `stdio` MCP servers.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports stdio MCP server transports.
    """
    http: Optional[McpHttpCapabilities] = None
    """
    Agent supports `http` MCP servers.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports HTTP MCP server transports.
    """
    acp: Optional[McpAcpCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Agent supports `acp` MCP servers.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports ACP MCP server transports.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("acp", "http", "stdio", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SessionSystemPromptModeCapabilities(BaseModel):
    override: Optional[SessionSystemPromptOverrideCapabilities] = None
    """
    Whether the Agent supports `override` mode.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports replacing its built-in default
    instructions with the Client-provided system prompt.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    Whether the Agent supports the `auth/status` method.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports the `auth/status` method.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("status", mode="wrap")
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
    Human-readable name of the authentication method, shown to users as plain text.
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
    If true, Clients MUST NOT call `providers/disable` for this provider ID.
    """
    current: Optional[ProviderCurrentConfig] = None
    """
    Current effective non-secret routing config.
    Null or omitted means provider is disabled.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("supported", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class AuthStatusResponse(BaseModel):
    state: Union[Literal["authenticated"], Literal["unauthenticated"], Literal["not_required"], Literal["unknown"], str]
    """
    Whether the Agent can start sessions without the user signing in first.

    Clients use this field to decide whether sign-in is needed. Clients MUST treat
    values they do not recognize as `unknown`.
    """
    accounts: Optional[List[AuthAccount]] = None
    """
    The accounts the Agent has configured, for display.

    Optional. Omitted and empty are equivalent and mean no accounts are reported.
    Agents that hold credentials for several providers or accounts SHOULD list each
    of them and mark the ones in effect with `active`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("accounts", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class BooleanSessionConfigOption(SessionConfigBoolean):
    config_id: Annotated[str, Field(alias="configId")]
    """
    Identifier for the configuration option. MUST be unique within `configOptions`.
    """
    name: str
    """
    Human-readable label the Client can display for the option, in plain text.
    """
    description: Optional[str] = None
    """
    Optional description for the Client to display to the user.
    May contain Markdown.

    Omitted and `null` are equivalent and mean no description is provided.
    """
    category: Optional[
        Union[
            Literal["mode"], Literal["model"], Literal["model_config"], Literal["speed"], Literal["thought_level"], str
        ]
    ] = None
    """
    Optional semantic category for this option (UX only).

    Omitted and `null` are equivalent and mean no category is provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    type: Literal["boolean"] = "boolean"


class OtherSessionConfigOption(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    config_id: Annotated[str, Field(alias="configId")]
    """
    Identifier for the configuration option. MUST be unique within `configOptions`.
    """
    name: str
    """
    Human-readable label the Client can display for the option, in plain text.
    """
    description: Optional[str] = None
    """
    Optional description for the Client to display to the user.
    May contain Markdown.

    Omitted and `null` are equivalent and mean no description is provided.
    """
    category: Optional[
        Union[
            Literal["mode"], Literal["model"], Literal["model_config"], Literal["speed"], Literal["thought_level"], str
        ]
    ] = None
    """
    Optional semantic category for this option (UX only).

    Omitted and `null` are equivalent and mean no category is provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
    Identifier for this group. MUST be unique within the option.
    """
    name: str
    """
    Human-readable label the Client can display as a plain-text group header.
    """
    options: Annotated[List[SessionConfigSelectOption], Field(min_length=1)]
    """
    The set of option values in this group. MUST contain at least one value.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class SkillCommand(Skill):
    type: Literal["skill"] = "skill"


class TextAvailableCommandInput(TextCommandInput):
    type: Literal["text"] = "text"


class Command(BaseModel):
    name: str
    """
    Plain-text command name without the leading `/`, such as `create_plan` or `research_codebase`.

    MUST NOT contain whitespace and MUST be unique within the command list.
    """
    description: str
    """
    Human-readable description of what the command does.
    May contain Markdown.
    """
    input: Optional[Union[TextAvailableCommandInput, OtherAvailableCommandInput]] = None
    """
    Input for the command if required

    Omitted and `null` are equivalent and mean the command takes no input.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("input", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ListSessionsResponse(BaseModel):
    sessions: List[SessionInfo]
    """
    Array of session information objects.

    Agents SHOULD return the sessions with the most recent activity first.
    """
    next_cursor: Annotated[Optional[str], Field(alias="nextCursor")] = None
    """
    Opaque cursor token. If present, pass this in the next request's cursor parameter
    to fetch the next page. Omitted and `null` are equivalent and mean there are no more
    results. An empty string is a valid cursor, not the end of results.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("sessions", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class TerminalBackgroundWork(BackgroundTerminal):
    type: Literal["terminal"] = "terminal"


class ToolCallBackgroundWork(BackgroundToolCall):
    type: Literal["tool_call"] = "tool_call"


class SubagentBackgroundWork(BackgroundSubagent):
    type: Literal["subagent"] = "subagent"


class RunningStateUpdate(BaseModel):
    background: Optional[
        List[Union[TerminalBackgroundWork, ToolCallBackgroundWork, SubagentBackgroundWork, OtherBackgroundWork]]
    ] = None
    """
    Work running independently of foreground work, as a complete snapshot.

    Optional. Omitted and `null` are equivalent and mean the Agent is not reporting
    background work, not that nothing is running. `[]` means no background work is
    running. An Agent that reports background work MUST include it on every state
    update for the session.

    See protocol docs: [Background Work](https://agentclientprotocol.com/protocol/v2/draft/prompt-lifecycle#background-work)
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("background", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class RequiresActionStateUpdate(BaseModel):
    background: Optional[
        List[Union[TerminalBackgroundWork, ToolCallBackgroundWork, SubagentBackgroundWork, OtherBackgroundWork]]
    ] = None
    """
    Work running independently of foreground work, as a complete snapshot.

    Optional. Omitted and `null` are equivalent and mean the Agent is not reporting
    background work, not that nothing is running. `[]` means no background work is
    running. An Agent that reports background work MUST include it on every state
    update for the session.

    See protocol docs: [Background Work](https://agentclientprotocol.com/protocol/v2/draft/prompt-lifecycle#background-work)
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("background", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class UnknownStateUpdate(BaseModel):
    background: Optional[
        List[Union[TerminalBackgroundWork, ToolCallBackgroundWork, SubagentBackgroundWork, OtherBackgroundWork]]
    ] = None
    """
    Work running independently of foreground work, as a complete snapshot.

    Optional. Omitted and `null` are equivalent and mean the Agent is not reporting
    background work, not that nothing is running. `[]` means no background work is
    running. An Agent that reports background work MUST include it on every state
    update for the session.

    See protocol docs: [Background Work](https://agentclientprotocol.com/protocol/v2/draft/prompt-lifecycle#background-work)
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("background", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class PromptResponse(BaseModel):
    message_id: Annotated[str, Field(alias="messageId")]
    """
    Identifies the user message inserted into the ACP conversation.

    Required and non-null. Omission and explicit `null` are both invalid.

    The corresponding user-message session update carries this same identifier and may arrive
    before or after this response. Agents MUST echo the message during the live session, but are
    not required to retain it. If retained and replayed, the message keeps this identifier.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class RunningSessionStateUpdateBase(RunningStateUpdate):
    state: Literal["running"] = "running"


class RequiresActionSessionStateUpdateBase(RequiresActionStateUpdate):
    state: Literal["requires_action"] = "requires_action"


class UnknownSessionStateUpdateBase(UnknownStateUpdate):
    state: Literal["unknown"] = "unknown"


class RunningSessionStateUpdate(RunningSessionStateUpdateBase, SessionStateUpdateBase):
    session_update: Annotated[Literal["state_update"], Field(alias="sessionUpdate")] = "state_update"


class RequiresActionSessionStateUpdate(RequiresActionSessionStateUpdateBase, SessionStateUpdateBase):
    session_update: Annotated[Literal["state_update"], Field(alias="sessionUpdate")] = "state_update"


class UnknownSessionStateUpdate(UnknownSessionStateUpdateBase, SessionStateUpdateBase):
    session_update: Annotated[Literal["state_update"], Field(alias="sessionUpdate")] = "state_update"


class SessionTerminalOutputChunk(TerminalOutputChunk):
    session_update: Annotated[Literal["terminal_output_chunk"], Field(alias="sessionUpdate")] = "terminal_output_chunk"


class SessionPlanRemovedUpdate(PlanRemoved):
    session_update: Annotated[Literal["plan_removed"], Field(alias="sessionUpdate")] = "plan_removed"


class SessionInfoUpdate(SessionInfoUpdateBase):
    session_update: Annotated[Literal["session_info_update"], Field(alias="sessionUpdate")] = "session_info_update"


class UsageUpdate(UsageUpdateBase):
    session_update: Annotated[Literal["usage_update"], Field(alias="sessionUpdate")] = "usage_update"


class SessionNotice(Notice):
    session_update: Annotated[Literal["notice"], Field(alias="sessionUpdate")] = "notice"


class SessionReplayComplete(ReplayComplete):
    session_update: Annotated[Literal["replay_complete"], Field(alias="sessionUpdate")] = "replay_complete"


class TerminalCapabilities(BaseModel):
    cancel: Optional[TerminalCancelCapabilities] = None
    """
    Permits the Client to cancel this terminal with `terminal/cancel`.

    Omitted and `null` are equivalent and mean unsupported. An object (including `{}`)
    means supported. Agents SHOULD stop advertising it once the terminal reports `exitStatus`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("cancel", mode="wrap")
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
    size: Optional[TerminalSize] = None
    """
    The terminal size the command runs with, when known.

    Clients can render output at this size so that column layouts, progress bars,
    and cursor-positioned output display as the program wrote them.
    """
    capabilities: Optional[TerminalCapabilities] = None
    """
    Client-initiated controls this terminal currently supports, such as cancellation.

    Omission leaves the capabilities unchanged, `null` clears them, and an object replaces
    the whole previous value. A new terminal starts with none.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted leaves the stored metadata unchanged. `null`
    clears it. An object patches the stored metadata by top-level key: keys it omits are
    unchanged, a key set to `null` is removed, and any other value replaces that key's
    stored value without merging nested objects. Beyond this patch rule, implementations
    MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Metadata Patches](https://agentclientprotocol.com/protocol/v2/draft/extensibility#metadata-patches)
    """

    @field_validator("capabilities", "command", "cwd", "exit_status", "output", "size", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


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
    May contain Markdown.
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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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

    The Agent MUST send all entries with their current status.
    The Client MUST replace the checklist with this `planId` on each update.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("entries", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class CursorCapabilities(BaseModel):
    fork: Optional[CursorForkCapabilities] = None
    """
    Permits the Client to fork the session at this cursor with `session/fork`.

    Omitted and `null` are equivalent and mean unsupported. An object (including `{}`)
    means supported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("fork", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class UserMessageCursorAnchor(CursorMessageAnchor):
    type: Literal["user_message"] = "user_message"


class AgentMessageCursorAnchor(CursorMessageAnchor):
    type: Literal["agent_message"] = "agent_message"


class AgentThoughtCursorAnchor(CursorMessageAnchor):
    type: Literal["agent_thought"] = "agent_thought"


class SessionMessageCursorAnchor(CursorMessageAnchor):
    type: Literal["session_message"] = "session_message"


class ToolCallCursorAnchor(CursorToolCallAnchor):
    type: Literal["tool_call"] = "tool_call"


class CompactionCursorAnchor(CursorCompactionAnchor):
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
        Union[
            UserMessageCursorAnchor,
            AgentMessageCursorAnchor,
            AgentThoughtCursorAnchor,
            SessionMessageCursorAnchor,
            ToolCallCursorAnchor,
            CompactionCursorAnchor,
            OtherCursorAnchor,
        ]
    ] = None
    """
    Where the cursor is, relative to an entity the Client holds.

    Required on the first update for a cursor. The Agent MUST NOT change it afterwards.
    A later update can omit it or repeat the same anchor. Omitted and `null` are
    equivalent and leave the stored anchor unchanged.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted leaves the stored metadata unchanged. `null`
    clears it. An object patches the stored metadata by top-level key: keys it omits are
    unchanged, a key set to `null` is removed, and any other value replaces that key's
    stored value without merging nested objects. Beyond this patch rule, implementations
    MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Metadata Patches](https://agentclientprotocol.com/protocol/v2/draft/extensibility#metadata-patches)
    """

    @field_validator("capabilities", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SubagentSessionCapabilities(BaseModel):
    cancel: Optional[SessionCancelCapabilities] = None
    """
    Permits the Client to cancel this child's current work without ending
    the session. Omitted and `null` are equivalent and mean unsupported. An object (including
    `{}`) means supported.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("cancel", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ClientAuthCapabilities(BaseModel):
    terminal: Optional[TerminalAuthCapabilities] = None
    """
    Whether the Client supports `terminal` authentication methods.

    Optional. Omitted and `null` are equivalent and mean the Client does not advertise support.
    Supplying `{}` means the Client can reproduce the configured Agent
    invocation in an interactive terminal, and the Agent MAY include
    `terminal` entries in its authentication methods.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("terminal", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ElicitationCapabilities(BaseModel):
    form: Optional[ElicitationFormCapabilities] = None
    """
    Whether the Client supports form-based elicitation.

    Optional. Omitted and `null` are equivalent and mean form support is not advertised.
    Supplying `{}` explicitly advertises form support.
    """
    url: Optional[ElicitationUrlCapabilities] = None
    """
    Whether the Client supports URL-based elicitation.

    Optional. Omitted and `null` are equivalent and mean the Client does not advertise support.
    Supplying `{}` means the Client supports URL-based elicitation.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("form", "url", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class PromptSuggestionsCapabilities(BaseModel):
    image: Optional[PromptSuggestionsImageCapabilities] = None
    """
    Whether prompt suggestions may include `image` content blocks.

    Optional. Omitted and `null` are equivalent and mean the Client does not advertise support.
    Supplying `{}` means the Client can offer suggestions that include images.
    """
    audio: Optional[PromptSuggestionsAudioCapabilities] = None
    """
    Whether prompt suggestions may include `audio` content blocks.

    Optional. Omitted and `null` are equivalent and mean the Client does not advertise support.
    Supplying `{}` means the Client can offer suggestions that include audio.
    """
    embedded_context: Annotated[
        Optional[PromptSuggestionsEmbeddedContextCapabilities], Field(alias="embeddedContext")
    ] = None
    """
    Whether prompt suggestions may include `resource` content blocks.

    Optional. Omitted and `null` are equivalent and mean the Client does not advertise support.
    Supplying `{}` means the Client can offer suggestions that include
    embedded resources.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("audio", "embedded_context", "image", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class CommandsCapabilities(BaseModel):
    skill: Optional[CommandsSkillCapabilities] = None
    """
    Support for `skill` command entries.

    Optional. Omitted and `null` are equivalent and mean the Client does not advertise support.
    Supplying `{}` means the Agent MAY send `skill` entries in `availableCommands`. The
    Client invokes them with `skill` content blocks.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("skill", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class HttpMcpServer(McpServerHttp):
    type: Literal["http"] = "http"


class AcpMcpServer(McpServerAcp):
    type: Literal["acp"] = "acp"


class StdioMcpServer(McpServerStdio):
    type: Literal["stdio"] = "stdio"


class ListSessionsRequest(BaseModel):
    cwd: Optional[str] = None
    """
    Filter sessions by working directory. Must be an absolute path.

    Only sessions whose `cwd` is this same directory match. Subdirectories and
    `additionalDirectories` don't. Agents MAY normalize paths when comparing.

    Omitted and `null` are equivalent and mean sessions are not filtered by working directory.
    Clients keep the same value when following `nextCursor`.
    """
    cursor: Optional[str] = None
    """
    Opaque cursor token from a previous response's nextCursor field for cursor-based pagination

    Omitted and `null` are equivalent and mean the first page is requested.
    """
    archived: Optional[Union[Literal["exclude"], Literal["include"], Literal["only"], str]] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Filter sessions by archive state.

    Optional. Omitted and `null` are equivalent and mean `exclude`. Only send this when
    the Agent advertises the `session.archive` capability. The filter combines with `cwd` and
    applies before pagination. Clients keep the same value when following `nextCursor`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ReplayFromStartVariant(ReplayFromStart):
    type: Literal["start"] = "start"


class CancelledPermissionOutcome(CancelledPermissionOutcomeBase):
    outcome: Literal["cancelled"] = "cancelled"


class SelectedPermissionOutcomeVariant(SelectedPermissionOutcome):
    outcome: Literal["selected"] = "selected"


class AcceptElicitationResponse(ElicitationAcceptAction):
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    action: Literal["accept"] = "accept"


class McpErrorResponse(BaseModel):
    error: McpError
    """
    Required, non-null MCP error object.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.
    This is ACP metadata, not the inner MCP message's `_meta`.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ClientNotification(MethodParamsModel):
    method: str
    """
    The notification method name.
    """
    params: Optional[Union[CancelSessionNotification, MessageMcpNotification, Any]] = None
    """
    Method-specific notification parameters.
    """


class ProxyNotification(MethodParamsModel):
    method: str
    """
    The notification method name.
    """
    params: Optional[ProxySuccessorNotification] = None
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
        SkillContentBlock,
        OtherContentBlock,
    ]
    """
    The actual content block.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ElicitationMultiSelectPropertySchema(MultiSelectPropertySchema):
    type: Literal["array"] = "array"


class PromptCapabilities(BaseModel):
    image: Optional[PromptImageCapabilities] = None
    """
    Agent supports `image` content blocks.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports image content in prompts.
    """
    audio: Optional[PromptAudioCapabilities] = None
    """
    Agent supports `audio` content blocks.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports audio content in prompts.
    """
    embedded_context: Annotated[Optional[PromptEmbeddedContextCapabilities], Field(alias="embeddedContext")] = None
    """
    Agent supports embedded context in `session/prompt` requests.

    When enabled, the Client is allowed to include `resource` content blocks
    in prompt requests for pieces of context that are referenced in the message.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports embedded context in prompts.
    """
    skill: Optional[PromptSkillCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Agent supports `skill` content blocks.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means a prompt may contain one `skill` content block, as its first block.
    An Agent that sends `skill` entries in `availableCommands` MUST advertise this capability.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("audio", "embedded_context", "image", "skill", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SessionSystemPromptCapabilities(BaseModel):
    mode: Optional[SessionSystemPromptModeCapabilities] = None
    """
    Delivery modes the Agent supports in addition to `append`.

    Optional. Omitted and `null` are equivalent and mean the Agent supports only `append`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class SessionConfigSelect(SessionConfigSelectModel):
    current_value: Annotated[str, Field(alias="currentValue")]
    """
    The currently selected value. MUST be one of the `value`s in `options`.
    """
    options: Union[List[SessionConfigSelectOption], List[SessionConfigSelectGroup]]
    """
    The set of selectable options. MUST contain at least one value, and each `value` MUST be
    unique within the option, including across groups.
    """


class SlashCommand(Command):
    type: Literal["command"] = "command"


class RunningState(RunningStateUpdate):
    state: Literal["running"] = "running"


class RequiresActionState(RequiresActionStateUpdate):
    state: Literal["requires_action"] = "requires_action"


class UnknownState(UnknownStateUpdate):
    state: Literal["unknown"] = "unknown"


class Error(ErrorModel):
    code: StrictInt
    """
    A number indicating the error type that occurred.
    This MUST be an integer as defined in the JSON-RPC specification.
    """
    message: str
    """
    A short, human-readable description of the error, in plain text.
    The sender SHOULD limit the message to a single concise sentence.
    """
    data: Any = None
    """
    Optional primitive or structured value that contains additional information about the error.
    This may include debugging information or context-specific details.

    Omitted and `null` are equivalent and mean no additional data is provided. Some codes
    define a shape: [`UnsupportedProtocolVersionErrorData`] for `-32022`, and
    [`SessionNotFoundErrorData`] for `-32002` when a session is not found.
    """

    @field_validator("data", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ErrorStopReason(BaseModel):
    error: Optional[Error] = None
    """
    The failure, as a JSON-RPC error object.

    Optional. Omitted and `null` are equivalent and mean the Agent is not reporting failure details.
    Agents SHOULD include it.
    """

    @field_validator("error", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class IdleStateUpdate(IdleStateUpdateModel):
    usage: Optional[Usage] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Token usage for completed foreground work.

    Optional. Omitted and `null` are equivalent and mean the Agent is not reporting token
    usage for this state update. A repeated or resume snapshot that carries it MUST NOT
    be counted as additional usage.
    """
    background: Optional[
        List[Union[TerminalBackgroundWork, ToolCallBackgroundWork, SubagentBackgroundWork, OtherBackgroundWork]]
    ] = None
    """
    Work running independently of foreground work, as a complete snapshot.

    Optional. Omitted and `null` are equivalent and mean the Agent is not reporting
    background work, not that nothing is running. `[]` means no background work is
    running. An Agent that reports background work MUST include it on every state
    update for the session.

    See protocol docs: [Background Work](https://agentclientprotocol.com/protocol/v2/draft/prompt-lifecycle#background-work)
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    stop_reason: Annotated[Optional[str], Field(alias="stopReason")] = None
    """
    Why foreground work stopped. The value selects one of this type's variants, which may add fields of their own.

    Optional. Omitted and `null` are equivalent and mean the Agent is not reporting a stop reason.

    See protocol docs: [Stop Reasons](https://agentclientprotocol.com/protocol/v2/draft/prompt-lifecycle#stop-reasons)
    """
    error: Optional[Error] = None
    """
    The failure, as a JSON-RPC error object.

    Optional. Omitted and `null` are equivalent and mean the Agent is not reporting failure details.
    Agents SHOULD include it.
    """

    @field_validator("background", "error", "stop_reason", "usage", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class IdleSessionStateUpdateBase(IdleStateUpdate):
    state: Literal["idle"] = "idle"


class IdleSessionStateUpdate(IdleSessionStateUpdateBase, SessionStateUpdateBase):
    session_update: Annotated[Literal["state_update"], Field(alias="sessionUpdate")] = "state_update"


class SessionTerminalUpdate(TerminalUpdate):
    session_update: Annotated[Literal["terminal_update"], Field(alias="sessionUpdate")] = "terminal_update"


class SessionCursorUpdate(CursorUpdate):
    session_update: Annotated[Literal["cursor_update"], Field(alias="sessionUpdate")] = "cursor_update"


class ContentChunk(BaseModel):
    message_id: Annotated[str, Field(alias="messageId")]
    """
    A unique identifier for the message this chunk belongs to.

    All chunks belonging to the same message share the same `messageId`.
    Chunks for different messages can interleave.
    """
    content: Union[
        TextContentBlock,
        ImageContentBlock,
        AudioContentBlock,
        ResourceContentBlock,
        EmbeddedResourceContentBlock,
        SkillContentBlock,
        OtherContentBlock,
    ]
    """
    A single item of content
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.
    This metadata is scoped to the chunk and does not patch stored metadata.

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
                SkillContentBlock,
                OtherContentBlock,
            ]
        ]
    ] = None
    """
    Complete replacement content for this message.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted leaves the stored metadata unchanged. `null`
    clears it. An object patches the stored metadata by top-level key: keys it omits are
    unchanged, a key set to `null` is removed, and any other value replaces that key's
    stored value without merging nested objects. Beyond this patch rule, implementations
    MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Metadata Patches](https://agentclientprotocol.com/protocol/v2/draft/extensibility#metadata-patches)
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
                SkillContentBlock,
                OtherContentBlock,
            ]
        ]
    ] = None
    """
    Complete replacement content for this message.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted leaves the stored metadata unchanged. `null`
    clears it. An object patches the stored metadata by top-level key: keys it omits are
    unchanged, a key set to `null` is removed, and any other value replaces that key's
    stored value without merging nested objects. Beyond this patch rule, implementations
    MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Metadata Patches](https://agentclientprotocol.com/protocol/v2/draft/extensibility#metadata-patches)
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
                SkillContentBlock,
                OtherContentBlock,
            ]
        ]
    ] = None
    """
    Complete replacement content for this thought message.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted leaves the stored metadata unchanged. `null`
    clears it. An object patches the stored metadata by top-level key: keys it omits are
    unchanged, a key set to `null` is removed, and any other value replaces that key's
    stored value without merging nested objects. Beyond this patch rule, implementations
    MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Metadata Patches](https://agentclientprotocol.com/protocol/v2/draft/extensibility#metadata-patches)
    """

    @field_validator("content", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class PlanUpdateItems(PlanItems):
    type: Literal["items"] = "items"


class PlanUpdate(BaseModel):
    plan: Union[PlanUpdateItems, PlanUpdateMarkdown, OtherPlanUpdateContent]
    """
    Replaces the whole plan with this `planId`, even if its `type` changes.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class AvailableCommandsUpdateBase(BaseModel):
    available_commands: Annotated[
        List[Union[SlashCommand, SkillCommand, OtherAvailableCommand]], Field(alias="availableCommands")
    ]
    """
    Commands the Agent can execute.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("available_commands", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class CompactionUpdate(BaseModel):
    compaction_id: Annotated[str, Field(alias="compactionId")]
    """
    The Agent-owned ID of this compaction, unique within the session.
    """
    status: Optional[
        Union[Literal["in_progress"], Literal["completed"], Literal["failed"], Literal["cancelled"], str]
    ] = None
    """
    Current lifecycle status.

    Omission leaves the status unchanged, `null` resets it to `in_progress`, and a value
    replaces it. A new compaction starts as `in_progress`.
    """
    summary: Optional[
        List[
            Union[
                TextContentBlock,
                ImageContentBlock,
                AudioContentBlock,
                ResourceContentBlock,
                EmbeddedResourceContentBlock,
                SkillContentBlock,
                OtherContentBlock,
            ]
        ]
    ] = None
    """
    Complete replacement user-displayable summary content for the compaction.
    """
    error: Optional[str] = None
    """
    Human-readable details of a compaction error, in plain text.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted leaves the stored metadata unchanged. `null`
    clears it. An object patches the stored metadata by top-level key: keys it omits are
    unchanged, a key set to `null` is removed, and any other value replaces that key's
    stored value without merging nested objects. Beyond this patch rule, implementations
    MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Metadata Patches](https://agentclientprotocol.com/protocol/v2/draft/extensibility#metadata-patches)
    """

    @field_validator("error", "status", mode="wrap")
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
        SkillContentBlock,
        OtherContentBlock,
    ]
    """
    One content block to append.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.
    This metadata is scoped to the chunk and does not patch stored metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class PromptSuggestion(BaseModel):
    prompt: Annotated[
        List[
            Union[
                TextContentBlock,
                ImageContentBlock,
                AudioContentBlock,
                ResourceContentBlock,
                EmbeddedResourceContentBlock,
                SkillContentBlock,
                OtherContentBlock,
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
    other content block types that both the Agent's `session.prompt`
    capability and the Client's `session.promptSuggestions` capability
    advertise. Agents MUST NOT send a prompt whose only content is empty
    or whitespace-only text.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("suggestions", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


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
            Union[
                TextContentBlock,
                ImageContentBlock,
                AudioContentBlock,
                ResourceContentBlock,
                EmbeddedResourceContentBlock,
                SkillContentBlock,
                OtherContentBlock,
            ]
        ]
    ] = None
    """
    Omitted leaves content unchanged. `null` clears it. A concrete array
    replaces the whole content collection.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted leaves the stored metadata unchanged. `null`
    clears it. An object patches the stored metadata by top-level key: keys it omits are
    unchanged, a key set to `null` is removed, and any other value replaces that key's
    stored value without merging nested objects. Beyond this patch rule, implementations
    MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Metadata Patches](https://agentclientprotocol.com/protocol/v2/draft/extensibility#metadata-patches)
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
    content: Union[
        TextContentBlock,
        ImageContentBlock,
        AudioContentBlock,
        ResourceContentBlock,
        EmbeddedResourceContentBlock,
        SkillContentBlock,
        OtherContentBlock,
    ]
    """
    A single content block appended to the message.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.
    This metadata is scoped to the chunk and does not patch stored metadata.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ClientSessionCapabilities(BaseModel):
    prompt_suggestions: Annotated[Optional[PromptSuggestionsCapabilities], Field(alias="promptSuggestions")] = None
    """
    Support for `prompt_suggestions_update` session updates.

    Optional. Omitted and `null` are equivalent and mean the Client does not advertise support.
    Supplying `{}` means the Client can offer prompt suggestions made of
    `text` and `resource_link` content blocks.
    """
    commands: Optional[CommandsCapabilities] = None
    """
    Command types the Client understands in addition to `command`.

    Optional. Omitted and `null` are equivalent and mean the Client does not advertise support
    for any of them. This object only groups capabilities. Supplying `{}` advertises nothing.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("commands", "prompt_suggestions", mode="wrap")
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
    content: Annotated[
        List[
            Union[
                TextContentBlock,
                ImageContentBlock,
                AudioContentBlock,
                ResourceContentBlock,
                EmbeddedResourceContentBlock,
                SkillContentBlock,
                OtherContentBlock,
            ]
        ],
        Field(min_length=1),
    ]
    """
    The system prompt content.

    Agents advertising the capability MUST accept `text` content blocks.
    Clients MUST NOT send other content block types until a future
    capability allows them, and Agents MUST return an error for content
    blocks they do not support, including unknown types.

    Clients MUST NOT send an empty array. Agents MAY reject one with
    `-32602` without creating or resuming the session.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ForkSessionRequest(InitialConfigRequestModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to fork.
    """
    cwd: str
    """
    The working directory for this session. Must be an absolute path.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = []
    """
    Additional workspace roots to activate for this session. Each path MUST be absolute.

    When omitted or empty, no additional roots are activated. When non-empty,
    this is the complete resulting additional-root list for the forked
    session.
    """
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, AcpMcpServer, StdioMcpServer, OtherMcpServer]]],
        Field(alias="mcpServers", validate_default=True),
    ] = []
    """
    List of MCP servers to connect to for this session.
    """
    system_prompt: Annotated[Optional[SystemPrompt], Field(alias="systemPrompt")] = None
    """
    System-level instructions for the forked session.

    Optional. Omitted and `null` are equivalent. The Agent MAY then keep a
    system prompt it stored with the source session or use its default
    instructions. Clients MUST only send this when the Agent advertises
    `session.systemPrompt`, and SHOULD send the same value they used for the
    source session.
    """
    config_options: Annotated[
        Optional[List[Union[InitialIdConfigOption, InitialBooleanConfigOption, OtherInitialConfigOption]]],
        Field(alias="configOptions", validate_default=True),
    ] = []
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Initial values for the forked session's config options.

    Optional. Omitted and `null` are equivalent to an empty list. Clients MUST only send
    this when the Agent advertises `session.options`.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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

    @field_validator("additional_directories", "config_options", "mcp_servers", mode="before")
    @classmethod
    def empty_list_on_null_validator(cls, v: Any, info: ValidationInfo) -> Any:
        return empty_list_on_null(v, info)


class ResumeSessionRequest(InitialConfigRequestModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to resume.
    """
    cwd: str
    """
    The working directory for this session. Must be an absolute path.

    Clients MUST send the session's working directory. Agents MAY reject a different directory
    with `-32602`, and MAY normalize paths when comparing.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = []
    """
    Additional workspace roots to activate for this session. Each path MUST be absolute.

    When omitted or empty, no additional roots are activated. When non-empty,
    this is the complete resulting additional-root list for the resumed
    session. It MAY differ from any previously used or reported list as long as
    the request `cwd` matches the session's `cwd`.
    """
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, AcpMcpServer, StdioMcpServer, OtherMcpServer]]],
        Field(alias="mcpServers", validate_default=True),
    ] = []
    """
    List of MCP servers to connect to for this session.

    The complete set of Client-provided servers. Omitted and `[]` mean none. Agents SHOULD
    NOT restore servers from earlier requests.
    """
    replay_from: Annotated[Optional[Union[ReplayFromStartVariant, OtherReplayFrom]], Field(alias="replayFrom")] = None
    """
    Inclusive cursor describing where conversation replay begins.

    Optional. Omitted and `null` are equivalent, and the Agent MUST NOT replay
    previous conversation history. Replay cursors are inclusive:
    replay includes the position identified by the cursor. Supplying
    `{ "type": "start" }` means the Agent MUST replay all retained
    conversation history before responding.

    Agents MUST reject unsupported cursor types or malformed cursor values
    with `-32602` (Invalid params), rather than substituting another cursor
    or resuming without replay.
    """
    system_prompt: Annotated[Optional[SystemPrompt], Field(alias="systemPrompt")] = None
    """
    System-level instructions for the resumed session.

    Optional. Omitted and `null` are equivalent. The Agent MAY then keep a
    system prompt it stored with the session or use its default instructions.
    Clients MUST only send this when the Agent advertises
    `session.systemPrompt`, and SHOULD send the same value they used when the
    session was created.
    """
    config_options: Annotated[
        Optional[List[Union[InitialIdConfigOption, InitialBooleanConfigOption, OtherInitialConfigOption]]],
        Field(alias="configOptions", validate_default=True),
    ] = []
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Initial values for the session's config options.

    Optional. Omitted and `null` are equivalent to an empty list. Clients MUST only send
    this when the Agent advertises `session.options`.

    The Agent applies the entries in list order, as if the Client had called
    `session/set_config_option` for each entry after the session was set up. Malformed
    entries and duplicate `configId` values fail the request with `-32602`. The Agent MUST
    ignore an entry whose `configId` is unknown, whose `type` does not match the option,
    or whose value is not allowed when it is applied. The `configOptions` in the response contain the values the session actually
    has. The Agent SHOULD NOT send `config_option_update` notifications for these entries
    on the requesting connection. If the session is active on other connections, the
    Agent SHOULD send `config_option_update` on those connections.

    The Agent applies the entries before it replays history for `replayFrom` and before
    it responds.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("additional_directories", "config_options", "mcp_servers", mode="before")
    @classmethod
    def empty_list_on_null_validator(cls, v: Any, info: ValidationInfo) -> Any:
        return empty_list_on_null(v, info)


class PromptRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The ID of the session to send this user message to
    """
    prompt: Annotated[
        List[
            Union[
                TextContentBlock,
                ImageContentBlock,
                AudioContentBlock,
                ResourceContentBlock,
                EmbeddedResourceContentBlock,
                SkillContentBlock,
                OtherContentBlock,
            ]
        ],
        Field(min_length=1),
    ]
    """
    The blocks of content that compose the user's message.

    As a baseline, the Agent MUST support `text` and `resource_link` content blocks,
    while other variants are optionally enabled via `PromptCapabilities`.

    The Client MUST adapt its interface according to `PromptCapabilities`.

    The Client MAY include referenced pieces of context as either
    `resource` or `resource_link` content blocks.

    When available, `resource` is preferred
    as it avoids extra round-trips and allows the message to include
    pieces of context from sources the Agent might not have access to.

    Clients MUST NOT send an empty array. Agents MAY reject one with `-32602` without
    inserting it.
    """
    idempotency_key: Annotated[Optional[str], Field(alias="idempotencyKey", max_length=255, min_length=1)] = None
    """
    Identifies one prompt submission within this session.

    Optional. Omitted and `null` are equivalent and mean no key. The Agent then handles
    the prompt without duplicate detection.

    The Client generates the key and sends the same key on every attempt for the same
    submission. Agents that advertise `capabilities.session` MUST support the key.
    A retry of a recorded insertion returns the original `messageId` without inserting
    another message.

    The Agent MUST recognize the key for as long as it retains the message, including
    across connections and Agent restarts. For a live-only message, it MUST recognize
    the key while the session stays active on a connection. A retry with a forgotten
    key is a new submission, even immediately after reconnecting. If a duplicate
    would matter and the Client cannot rule it out, it SHOULD NOT retry automatically.

    When present, the value is an opaque, case-sensitive string of 1 to 255 characters.
    Agents MUST reject any other value with `-32602` (Invalid params) and MUST NOT treat
    it as omitted.

    See protocol docs: [Retrying a prompt](https://agentclientprotocol.com/protocol/v2/draft/prompt-lifecycle#retrying-a-prompt)
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ClientErrorMessage(BaseModel):
    id: Optional[Union[int, str]]
    """
    The id of the request this response answers.

    Required. `null` only when the request's id could not be determined, such as
    for a parse error or an invalid request.
    """
    error: Error
    """
    Method-specific error data.
    """


class RequestPermissionResponse(BaseModel):
    outcome: Union[CancelledPermissionOutcome, SelectedPermissionOutcomeVariant, OtherPermissionOutcome]
    """
    The user's decision on the permission request.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ProxyErrorMessage(BaseModel):
    id: Optional[Union[int, str]]
    """
    The id of the request this response answers.

    Required. `null` only when the request's id could not be determined, such as
    for a parse error or an invalid request.
    """
    error: Error
    """
    Method-specific error data.
    """


class ContentToolCallContent(Content):
    type: Literal["content"] = "content"


class ElicitationSchema(BaseModel):
    type: Literal["object"] = "object"
    """
    Type discriminator. Always `"object"`. Any other value is rejected.
    """
    title: Optional[str] = None
    """
    Optional plain-text title for the form as a whole.

    Optional. Omitted and `null` are equivalent and mean no title is provided.
    """
    properties: Dict[
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
    Optional plain-text description of what the form asks the user to provide.

    Optional. Omitted and `null` are equivalent and mean no schema description is provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("description", "title", mode="wrap")
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
    Form-based elicitation mode where the Client renders a form from the provided schema.
    """


class AgentErrorMessage(BaseModel):
    id: Optional[Union[int, str]]
    """
    The id of the request this response answers.

    Required. `null` only when the request's id could not be determined, such as
    for a parse error or an invalid request.
    """
    error: Error
    """
    Method-specific error data.
    """


class SessionCapabilities(BaseModel):
    prompt: Optional[PromptCapabilities] = None
    """
    Prompt capabilities supported by the Agent in `session/prompt` requests.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise any
    prompt extensions beyond the baseline text and resource-link content
    required by `session/prompt`.
    """
    mcp: Optional[McpCapabilities] = None
    """
    MCP capabilities supported by the Agent for session lifecycle requests.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise MCP
    server transport support for sessions.
    """
    delete: Optional[SessionDeleteCapabilities] = None
    """
    Whether the Agent supports `session/delete`.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports deleting sessions from `session/list`.
    """
    additional_directories: Annotated[
        Optional[SessionAdditionalDirectoriesCapabilities], Field(alias="additionalDirectories")
    ] = None
    """
    Whether the Agent supports `additionalDirectories` on supported session lifecycle requests.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports `additionalDirectories` on
    supported session lifecycle requests.

    Agents MAY return `SessionInfo.additionalDirectories` to report the
    complete ordered additional-root list associated with a listed session.
    """
    system_prompt: Annotated[Optional[SessionSystemPromptCapabilities], Field(alias="systemPrompt")] = None
    """
    Whether the Agent supports `systemPrompt` on session lifecycle requests.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports `systemPrompt` in `append` mode.
    """
    fork: Optional[SessionForkCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the Agent supports `session/fork`.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports forking sessions.
    """
    archive: Optional[SessionArchiveCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the Agent supports `session/archive` and `session/unarchive`.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports both methods, the `archived` filter on
    `session/list`, and `SessionInfo.archived`.
    """
    status: Optional[SessionStatusCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the Agent supports `session/status`.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports checking session status.
    """
    options: Optional[SessionOptionsCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the Agent supports `session/options`.

    Optional. Omitted or `null` both mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports `session/options` and accepts initial
    config option values on session setup requests.
    """
    title: Optional[SessionTitleCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Whether the Agent supports `session/set_title`.

    Agents report titles through `session/list` and `session_info_update` without this
    capability. It only advertises that Clients can set titles.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports setting session titles.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator(
        "additional_directories",
        "archive",
        "delete",
        "fork",
        "mcp",
        "options",
        "prompt",
        "status",
        "system_prompt",
        "title",
        mode="wrap",
    )
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class SelectSessionConfigOption(SessionConfigSelect):
    config_id: Annotated[str, Field(alias="configId")]
    """
    Identifier for the configuration option. MUST be unique within `configOptions`.
    """
    name: str
    """
    Human-readable label the Client can display for the option, in plain text.
    """
    description: Optional[str] = None
    """
    Optional description for the Client to display to the user.
    May contain Markdown.

    Omitted and `null` are equivalent and mean no description is provided.
    """
    category: Optional[
        Union[
            Literal["mode"], Literal["model"], Literal["model_config"], Literal["speed"], Literal["thought_level"], str
        ]
    ] = None
    """
    Optional semantic category for this option (UX only).

    Omitted and `null` are equivalent and mean no category is provided.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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

    Optional. Omitted or empty means no config options are advertised.
    Senders MUST use an array, not `null`.
    Later `config_option_update` notifications replace this list.
    """
    available_commands: Annotated[
        Optional[List[Union[SlashCommand, SkillCommand, OtherAvailableCommand]]], Field(alias="availableCommands")
    ] = None
    """
    Initial commands the Agent can execute in this session.

    Optional. Omitted or empty means no initial commands are advertised.
    Senders MUST use an array, not `null`.
    Later `available_commands_update` notifications replace this list.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("available_commands", "config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class IdleState(IdleStateUpdate):
    state: Literal["idle"] = "idle"


class SessionOptionsResponse(BaseModel):
    config_options: Annotated[
        Optional[List[Union[SelectSessionConfigOption, BooleanSessionConfigOption, OtherSessionConfigOption]]],
        Field(alias="configOptions"),
    ] = None
    """
    The config options the session would start with, with the same rules as on
    `NewSessionResponse`.

    Optional. Omitted or empty means no config options are advertised.
    Senders MUST use an array, not `null`.
    """
    available_commands: Annotated[
        Optional[List[Union[SlashCommand, SkillCommand, OtherAvailableCommand]]], Field(alias="availableCommands")
    ] = None
    """
    Commands the Agent knows the session would have.

    Optional. Omitted or empty means no commands are advertised.
    Senders MUST use an array, not `null`.

    Advisory. Agents MAY return a partial list. Clients MUST use the commands from the
    `session/new` or `session/resume` response and later `available_commands_update`
    notifications for the session itself.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("available_commands", "config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class ResumeSessionResponse(BaseModel):
    config_options: Annotated[
        Optional[List[Union[SelectSessionConfigOption, BooleanSessionConfigOption, OtherSessionConfigOption]]],
        Field(alias="configOptions"),
    ] = None
    """
    Current session configuration options.

    Optional. Omitted or empty means no config options are advertised.
    Senders MUST use an array, not `null`.
    Later `config_option_update` notifications replace this list.
    """
    available_commands: Annotated[
        Optional[List[Union[SlashCommand, SkillCommand, OtherAvailableCommand]]], Field(alias="availableCommands")
    ] = None
    """
    Current commands the Agent can execute in this session.

    Optional. Omitted or empty means no initial commands are advertised.
    Senders MUST use an array, not `null`.
    Later `available_commands_update` notifications replace this list.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("available_commands", "config_options", mode="wrap")
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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


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


class SessionCompactionUpdate(CompactionUpdate):
    session_update: Annotated[Literal["compaction_update"], Field(alias="sessionUpdate")] = "compaction_update"


class SessionCompactionSummaryChunk(CompactionSummaryChunk):
    session_update: Annotated[Literal["compaction_summary_chunk"], Field(alias="sessionUpdate")] = (
        "compaction_summary_chunk"
    )


class SessionPromptSuggestionsUpdate(PromptSuggestionsUpdate):
    session_update: Annotated[Literal["prompt_suggestions_update"], Field(alias="sessionUpdate")] = (
        "prompt_suggestions_update"
    )


class SessionMessageUpdate(SessionMessage):
    session_update: Annotated[Literal["session_message"], Field(alias="sessionUpdate")] = "session_message"


class SessionMessageChunkUpdate(SessionMessageChunk):
    session_update: Annotated[Literal["session_message_chunk"], Field(alias="sessionUpdate")] = "session_message_chunk"


class ToolCallContentChunk(BaseModel):
    tool_call_id: Annotated[str, Field(alias="toolCallId")]
    """
    The ID of the tool call this content belongs to.
    """
    content: Union[
        ContentToolCallContent,
        DiffToolCallContent,
        TerminalToolCallContent,
        SessionToolCallContent,
        OtherToolCallContent,
    ]
    """
    A single item of content produced by the tool call.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.
    This metadata is scoped to the chunk and does not patch stored metadata.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class SubagentUpdate(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The opaque session ID identifying the child in all ACP messages.
    """
    title: Optional[str] = None
    """
    The parent's human-readable display title for this child, in plain text. It need not be unique.

    Optional and nullable. Omitted means unchanged. `null` clears it. If no
    title is set, the Client chooses a fallback presentation.
    """
    description: Optional[str] = None
    """
    The parent's human-readable description of the child's role or purpose.
    May contain Markdown.

    Optional and nullable. Omitted means unchanged. `null` clears it. This is
    current display metadata, not the history of instructions sent to the child.
    """
    capabilities: Optional[SubagentSessionCapabilities] = None
    """
    Client-initiated session mutations permitted for this subagent session.

    Read-only operations retain their normal protocol semantics and
    capability requirements.
    """
    state: Optional[Union[RunningState, IdleState, RequiresActionState, UnknownState, OtherState]] = None
    """
    The child's current foreground state, mirrored onto its parent association.

    Optional and nullable. Omitted means unchanged. `null` removes the
    current report and leaves activity unconfirmed (not idle or cancelled).
    A concrete `StateUpdate` replaces the entire previous snapshot.
    """
    cwd: Optional[str] = None
    """
    The child's working directory. Must be an absolute path.

    Optional and nullable. Omitted means unchanged. `null` clears the
    reported value. While unset, the child inherits its immediate parent's
    effective working directory.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = None
    """
    Additional workspace roots for the child. Each path MUST be absolute.

    Optional and nullable. Omitted means unchanged. `null` clears the
    reported value. While unset, the child inherits its immediate parent's
    effective additional directories, independently of `cwd`. A concrete
    array replaces the whole list. `[]` reports no additional directories.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted leaves the stored metadata unchanged. `null`
    clears it. An object patches the stored metadata by top-level key: keys it omits are
    unchanged, a key set to `null` is removed, and any other value replaces that key's
    stored value without merging nested objects. Beyond this patch rule, implementations
    MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Metadata Patches](https://agentclientprotocol.com/protocol/v2/draft/extensibility#metadata-patches)
    """

    @field_validator("capabilities", "description", "state", "title", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ClientCapabilities(BaseModel):
    auth: Optional[ClientAuthCapabilities] = None
    """
    Authentication capabilities supported by the Client.
    Determines which authentication method types the Agent can include
    in its `InitializeResponse`.

    Optional. Omitted and `null` are equivalent and mean the Client does not advertise any
    authentication-method extensions.
    """
    elicitation: Optional[ElicitationCapabilities] = None
    """
    Elicitation capabilities supported by the Client.
    Determines which elicitation modes the Agent can use.

    Optional. Omitted and `null` are equivalent and mean the Client does not advertise
    elicitation support.
    """
    session: Optional[ClientSessionCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Session-related capabilities supported by the Client.

    Optional. Omitted and `null` are equivalent and mean the Client does not advertise any
    session-related capabilities. This object only groups capabilities. Supplying `{}`
    advertises nothing.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("auth", "elicitation", "session", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class NewSessionRequest(InitialConfigRequestModel):
    cwd: str
    """
    The working directory for this session. Must be an absolute path.
    """
    additional_directories: Annotated[Optional[List[str]], Field(alias="additionalDirectories")] = []
    """
    Additional workspace roots for this session. Each path MUST be absolute.

    These expand the session's workspace scope without changing `cwd`, which
    remains the base for relative paths. When omitted or empty, no
    additional roots are activated for the new session.
    """
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, AcpMcpServer, StdioMcpServer, OtherMcpServer]]],
        Field(alias="mcpServers", validate_default=True),
    ] = []
    """
    List of MCP (Model Context Protocol) servers for the Agent to connect to.
    """
    system_prompt: Annotated[Optional[SystemPrompt], Field(alias="systemPrompt")] = None
    """
    System-level instructions for the new session.

    Optional. Omitted and `null` are equivalent and mean the Agent uses its
    own default instructions. Clients MUST only send this when the Agent
    advertises `session.systemPrompt`.
    """
    config_options: Annotated[
        Optional[List[Union[InitialIdConfigOption, InitialBooleanConfigOption, OtherInitialConfigOption]]],
        Field(alias="configOptions", validate_default=True),
    ] = []
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Initial values for the session's config options.

    Optional. Omitted and `null` are equivalent to an empty list. Clients MUST only send
    this when the Agent advertises `session.options`.

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
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("additional_directories", "config_options", "mcp_servers", mode="before")
    @classmethod
    def empty_list_on_null_validator(cls, v: Any, info: ValidationInfo) -> Any:
        return empty_list_on_null(v, info)


class ClientResponseMessage(BaseModel):
    id: Union[int, str]
    """
    The id of the request this response answers.
    """
    result: Union[
        RequestPermissionResponse,
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
    Unique identifier for this tool call within the session.
    """
    name: Optional[str] = None
    """
    Programmatic name of the tool being invoked.

    This field is optional and has patch semantics. Omission means no
    change, `null` clears the name, and a string replaces it. For a tool
    call ID the Client has not seen before, omission or `null` means that no
    tool name is available.
    """
    title: Optional[str] = None
    """
    Human-readable plain-text title describing what the tool is doing.

    Omission leaves the title unchanged, `null` unsets it, and a string replaces it.
    A new tool call starts without a title.
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
    Helps Clients choose appropriate icons and UI treatment.

    Omission leaves the kind unchanged, `null` resets it to `other`, and a value replaces it.
    A new tool call starts as `other`.
    """
    status: Optional[
        Union[
            Literal["pending"],
            Literal["in_progress"],
            Literal["paused"],
            Literal["completed"],
            Literal["failed"],
            Literal["cancelled"],
            str,
        ]
    ] = None
    """
    Current execution status of the tool call.

    Omission leaves the status unchanged, `null` resets it to `pending`, and a value
    replaces it. A new tool call starts as `pending`. Agents do not move a tool call
    back to `pending` once it has another status.
    """
    content: Optional[
        List[
            Union[
                ContentToolCallContent,
                DiffToolCallContent,
                TerminalToolCallContent,
                SessionToolCallContent,
                OtherToolCallContent,
            ]
        ]
    ] = None
    """
    Content produced by the tool call.
    """
    locations: Optional[List[ToolCallLocation]] = None
    """
    File locations affected by this tool call.
    Enables "follow-along" features in Clients.
    """
    raw_input: Annotated[Optional[Any], Field(alias="rawInput")] = None
    """
    Raw input parameters sent to the tool.
    """
    raw_output: Annotated[Optional[Any], Field(alias="rawOutput")] = None
    """
    Raw output returned by the tool.
    """
    capabilities: Optional[ToolCallCapabilities] = None
    """
    Client-initiated controls this tool call currently supports, such as cancellation.

    Omission leaves the capabilities unchanged, `null` clears them, and an object replaces
    the whole previous value. A new tool call starts with none.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted leaves the stored metadata unchanged. `null`
    clears it. An object patches the stored metadata by top-level key: keys it omits are
    unchanged, a key set to `null` is removed, and any other value replaces that key's
    stored value without merging nested objects. Beyond this patch rule, implementations
    MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Metadata Patches](https://agentclientprotocol.com/protocol/v2/draft/extensibility#metadata-patches)
    """

    @field_validator("capabilities", "kind", "name", "raw_input", "raw_output", "status", "title", mode="wrap")
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

    Clients apply this to their stored tool call exactly like a `tool_call_update`
    notification, in the order it was received, before showing the prompt. A first-seen
    `toolCallId` creates the tool call.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
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
    A plain-text message explaining to the user what the Agent needs and why.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    mode: Literal["form"] = "form"


class CreateFormRequestElicitationRequest(CreateFormRequestElicitationRequestBase, CreateFormElicitationRequestBase):
    message: str
    """
    A plain-text message explaining to the user what the Agent needs and why.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """
    mode: Literal["form"] = "form"


class AgentCapabilities(BaseModel):
    session: Optional[SessionCapabilities] = None
    """
    Session capabilities supported by the Agent.

    Optional. Omitted and `null` are equivalent and mean the Agent does not support the
    `session/*` methods. Supplying `{}` means the Agent supports the
    baseline session methods: `session/new`, `session/list`,
    `session/resume`, `session/close`, `session/prompt`, `session/cancel`,
    and `session/update`.
    """
    auth: Optional[AgentAuthCapabilities] = None
    """
    Authentication-related extension capabilities supported by the Agent.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise any
    authentication-related extensions. This field does not advertise support
    for `auth/login` or `auth/logout`. Those methods are advertised by a
    non-empty `authMethods` list in the `initialize` response.
    """
    providers: Optional[ProvidersCapabilities] = None
    """
    **UNSTABLE**

    This capability is not part of the spec yet, and may be removed or changed at any point.

    Provider configuration capabilities supported by the Agent.

    Optional. Omitted and `null` are equivalent and mean the Agent does not advertise support.
    Supplying `{}` means the Agent supports provider configuration methods.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("auth", "providers", "session", mode="wrap")
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

    Optional. Omitted or empty means no config options are advertised.
    Senders MUST use an array, not `null`.
    Later `config_option_update` notifications replace this list.
    """
    available_commands: Annotated[
        Optional[List[Union[SlashCommand, SkillCommand, OtherAvailableCommand]]], Field(alias="availableCommands")
    ] = None
    """
    Initial commands the Agent can execute in this session.

    Optional. Omitted or empty means no initial commands are advertised.
    Senders MUST use an array, not `null`.
    Later `available_commands_update` notifications replace this list.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("available_commands", "config_options", mode="wrap")
    @classmethod
    def skip_invalid_items_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return skip_invalid_items(v, handler, info)


class SessionStatusResponse(BaseModel):
    status: Union[Literal["open"], Literal["saved"], str]
    """
    Whether the Agent has the session in memory.

    Required. Clients MUST treat a value they do not recognize as `saved`.
    """
    state: Optional[Union[RunningState, IdleState, RequiresActionState, UnknownState, OtherState]] = None
    """
    The session's current foreground state, as the same snapshot the Agent sends in
    `state_update` notifications.

    Optional. Agents SHOULD include it when `status` is `open`. Omitted and `null`
    mean the Agent is not reporting foreground state, and Clients
    MUST NOT read that as `idle`. This is a snapshot, not a new transition. Clients
    MUST NOT count an `idle` snapshot as a new completion.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("state", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class ToolCallContentChunkUpdate(ToolCallContentChunk):
    session_update: Annotated[Literal["tool_call_content_chunk"], Field(alias="sessionUpdate")] = (
        "tool_call_content_chunk"
    )


class SessionToolCallUpdate(ToolCallUpdate):
    session_update: Annotated[Literal["tool_call_update"], Field(alias="sessionUpdate")] = "tool_call_update"


class ConfigOptionUpdate(ConfigOptionUpdateBase):
    session_update: Annotated[Literal["config_option_update"], Field(alias="sessionUpdate")] = "config_option_update"


class SessionSubagentUpdate(SubagentUpdate):
    session_update: Annotated[Literal["subagent_update"], Field(alias="sessionUpdate")] = "subagent_update"


class InitializeRequest(BaseModel):
    protocol_version: Annotated[int, Field(alias="protocolVersion", ge=0, le=65535)]
    """
    The latest protocol version supported by the Client.
    """
    info: Implementation
    """
    Information about the implementation sending this initialize request.
    """
    capabilities: Annotated[Optional[ClientCapabilities], Field(validate_default=True)] = {}
    """
    Capabilities supported by the Client.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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


class InitializeProxyRequest(InitializeRequest):
    pass


class ToolCallPermissionSubjectVariant(ToolCallPermissionSubject):
    type: Literal["tool_call"] = "tool_call"


class InitializeResponse(BaseModel):
    protocol_version: Annotated[int, Field(alias="protocolVersion", ge=0, le=65535)]
    """
    The highest protocol version supported by the Agent that is at or below
    the version requested by the Client. If no such version exists,
    initialization fails with error code -32022.

    The Client MUST check that it supports this version and SHOULD disconnect
    if it does not.
    """
    info: Implementation
    """
    Information about the implementation sending this initialize response.
    """
    capabilities: Annotated[Optional[AgentCapabilities], Field(validate_default=True)] = {}
    """
    Capabilities supported by the Agent.
    """
    auth_methods: Annotated[
        Optional[List[Union[TerminalAuthMethod, AgentAuthMethod, OtherAuthMethod]]], Field(alias="authMethods")
    ] = None
    """
    Authentication methods supported by the Agent.

    Optional. Omitted or empty means the Agent does not advertise the
    `auth/login` and `auth/logout` methods. Supplying one or more methods means
    the Agent MUST support both `auth/login` and `auth/logout`.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

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
        SessionNotice,
        SessionCompactionUpdate,
        SessionCompactionSummaryChunk,
        SessionReplayComplete,
        SessionPromptSuggestionsUpdate,
        SessionCursorUpdate,
        SessionSubagentUpdate,
        SessionMessageUpdate,
        SessionMessageChunkUpdate,
        OtherSessionUpdate,
        RunningSessionStateUpdate,
        IdleSessionStateUpdate,
        RequiresActionSessionStateUpdate,
        UnknownSessionStateUpdate,
        OtherSessionStateUpdate,
    ]
    """
    The actual update content.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """


class ClientRequest(MethodParamsModel):
    id: Union[int, str]
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
            AuthStatusRequest,
            NewSessionRequest,
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
            CancelToolCallRequest,
            CancelTerminalRequest,
            PromptRequest,
            SetSessionConfigOptionIdRequest,
            SetSessionConfigOptionBooleanRequest,
            SetSessionConfigOptionOtherRequest,
            Any,
        ]
    ] = None
    """
    Method-specific request parameters.
    """


class ProxyRequest(MethodParamsModel):
    id: Union[int, str]
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


class RequestPermissionRequest(BaseModel):
    session_id: Annotated[str, Field(alias="sessionId")]
    """
    The session ID for this request.
    """
    title: str
    """
    Human-readable title shown with the permission prompt, in plain text.

    This title is specific to the permission prompt and does not update any
    subject's displayed title.
    """
    description: Optional[str] = None
    """
    Optional human-readable explanation of why permission is needed.
    May contain Markdown.

    This text is specific to the permission prompt and does not update any
    subject's displayed content. Omitted and `null` are equivalent and mean no separate
    permission description was provided.
    """
    subject: Optional[
        Union[ToolCallPermissionSubjectVariant, CommandPermissionSubjectVariant, OtherPermissionSubject]
    ] = None
    """
    Optional structured context about the operation requiring permission.

    Omitted and `null` are equivalent and mean no structured subject was provided.
    """
    options: Annotated[List[PermissionOption], Field(min_length=1)]
    """
    Available permission options for the user to choose from.
    Must contain at least one option.
    """
    field_meta: Annotated[Optional[Dict[str, Any]], Field(alias="_meta")] = None
    """
    The _meta property is reserved by ACP to allow Clients and Agents to attach additional
    metadata to their interactions. Omitted and `null` are equivalent and mean no metadata.
    Implementations MUST NOT assign meaning to keys they do not recognize.

    See protocol docs: [Extensibility](https://agentclientprotocol.com/protocol/v2/draft/extensibility)
    """

    @field_validator("description", mode="wrap")
    @classmethod
    def use_default_on_error_validator(cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        return use_default_on_error(v, handler, info)


class AgentResponseMessage(BaseModel):
    id: Union[int, str]
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
        AuthStatusResponse,
        NewSessionResponse,
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
        CancelToolCallResponse,
        CancelTerminalResponse,
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


class AgentNotification(MethodParamsModel):
    method: str
    """
    The notification method name.
    """
    params: Optional[
        Union[UpdateSessionNotification, CompleteElicitationNotification, AuthStatusUpdateNotification, Any]
    ] = None
    """
    Method-specific notification parameters.
    """


class ProxyResponseMessage(BaseModel):
    id: Union[int, str]
    """
    The id of the request this response answers.
    """
    result: Any
    """
    Method-specific response data.
    """


class ProxyResponse(RootModel[Union[ProxyResponseMessage, ProxyErrorMessage]]):
    root: Union[ProxyResponseMessage, ProxyErrorMessage]
    """
    A JSON-RPC response object.
    """


class AgentRequest(MethodParamsModel):
    id: Union[int, str]
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


MessageMcpResponse = Annotated[Union[McpResultResponse, McpErrorResponse], Field(union_mode="left_to_right")]
IdempotencyKey = Annotated[str, Field(min_length=1, max_length=255)]
SetSessionConfigOptionRequest = (
    SetSessionConfigOptionIdRequest | SetSessionConfigOptionBooleanRequest | SetSessionConfigOptionOtherRequest
)
CreateElicitationRequest = (
    CreateOtherSessionElicitationRequest
    | CreateOtherRequestElicitationRequest
    | CreateFormSessionElicitationRequest
    | CreateFormRequestElicitationRequest
    | CreateUrlSessionElicitationRequest
    | CreateUrlRequestElicitationRequest
)
CreateElicitationResponse = (
    AcceptElicitationResponse | DeclineElicitationResponse | CancelElicitationResponse | OtherElicitationResponse
)
ProxySuccessorResponse = Any
