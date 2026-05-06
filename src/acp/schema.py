# Generated from schema/schema.json. Do not edit by hand.
# Schema ref: refs/tags/v0.12.2

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import AnyUrl, BaseModel as _BaseModel, Field, RootModel, ConfigDict, field_validator

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


class Jsonrpc(Enum):
    field_2_0 = "2.0"


class AuthCapabilities(BaseModel):
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
    # Whether the client supports `terminal` authentication methods.
    #
    # When `true`, the agent may include `terminal` entries in its authentication methods.
    terminal: Annotated[
        Optional[bool],
        Field(
            description="Whether the client supports `terminal` authentication methods.\n\nWhen `true`, the agent may include `terminal` entries in its authentication methods."
        ),
    ] = False


class AuthEnvVar(BaseModel):
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
    # Human-readable label for this variable, displayed in client UI.
    label: Annotated[
        Optional[str],
        Field(description="Human-readable label for this variable, displayed in client UI."),
    ] = None
    # The environment variable name (e.g. `"OPENAI_API_KEY"`).
    name: Annotated[
        str,
        Field(description='The environment variable name (e.g. `"OPENAI_API_KEY"`).'),
    ]
    # Whether this variable is optional.
    #
    # Defaults to `false`.
    optional: Annotated[
        Optional[bool],
        Field(description="Whether this variable is optional.\n\nDefaults to `false`."),
    ] = False
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


class AuthMethodAgent(BaseModel):
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
    # Optional description providing more details about this authentication method.
    description: Annotated[
        Optional[str],
        Field(description="Optional description providing more details about this authentication method."),
    ] = None
    # Unique identifier for this authentication method.
    id: Annotated[str, Field(description="Unique identifier for this authentication method.")]
    # Human-readable name of the authentication method.
    name: Annotated[str, Field(description="Human-readable name of the authentication method.")]


class AuthMethodEnvVar(BaseModel):
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
    # Optional description providing more details about this authentication method.
    description: Annotated[
        Optional[str],
        Field(description="Optional description providing more details about this authentication method."),
    ] = None
    # Unique identifier for this authentication method.
    id: Annotated[str, Field(description="Unique identifier for this authentication method.")]
    # Optional link to a page where the user can obtain their credentials.
    link: Annotated[
        Optional[str],
        Field(description="Optional link to a page where the user can obtain their credentials."),
    ] = None
    # Human-readable name of the authentication method.
    name: Annotated[str, Field(description="Human-readable name of the authentication method.")]
    # The environment variables the client should set.
    vars: Annotated[
        List[AuthEnvVar],
        Field(description="The environment variables the client should set."),
    ]


class AuthMethodTerminal(BaseModel):
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
    # Additional arguments to pass when running the agent binary for terminal auth.
    args: Annotated[
        Optional[List[str]],
        Field(description="Additional arguments to pass when running the agent binary for terminal auth."),
    ] = None
    # Optional description providing more details about this authentication method.
    description: Annotated[
        Optional[str],
        Field(description="Optional description providing more details about this authentication method."),
    ] = None
    # Additional environment variables to set when running the agent binary for terminal auth.
    env: Annotated[
        Optional[Dict[str, str]],
        Field(description="Additional environment variables to set when running the agent binary for terminal auth."),
    ] = None
    # Unique identifier for this authentication method.
    id: Annotated[str, Field(description="Unique identifier for this authentication method.")]
    # Human-readable name of the authentication method.
    name: Annotated[str, Field(description="Human-readable name of the authentication method.")]


class AuthenticateRequest(BaseModel):
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
    # The ID of the authentication method to use.
    # Must be one of the methods advertised in the initialize response.
    method_id: Annotated[
        str,
        Field(
            alias="methodId",
            description="The ID of the authentication method to use.\nMust be one of the methods advertised in the initialize response.",
        ),
    ]


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


class BlobResourceContents(BaseModel):
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
    blob: str
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    uri: str


class BooleanPropertySchema(BaseModel):
    # Default value.
    default: Annotated[Optional[bool], Field(description="Default value.")] = None
    # Human-readable description.
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    # Optional title for the property.
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None


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


class Cost(BaseModel):
    # Total cumulative cost for session.
    amount: Annotated[float, Field(description="Total cumulative cost for session.")]
    # ISO 4217 currency code (e.g., "USD", "EUR").
    currency: Annotated[str, Field(description='ISO 4217 currency code (e.g., "USD", "EUR").')]


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


class CreateTerminalResponse(BaseModel):
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
    # The unique identifier for the created terminal.
    terminal_id: Annotated[
        str,
        Field(
            alias="terminalId",
            description="The unique identifier for the created terminal.",
        ),
    ]


class Diff(BaseModel):
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
    # The new content after modification.
    new_text: Annotated[str, Field(alias="newText", description="The new content after modification.")]
    # The original content (None for new files).
    old_text: Annotated[
        Optional[str],
        Field(alias="oldText", description="The original content (None for new files)."),
    ] = None
    # The file path being modified.
    path: Annotated[str, Field(description="The file path being modified.")]


class DisableProvidersRequest(BaseModel):
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
    # Provider id to disable.
    id: Annotated[str, Field(description="Provider id to disable.")]


class DisableProvidersResponse(BaseModel):
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


class ElicitationAcceptAction(BaseModel):
    # The user-provided content, if any, as an object matching the requested schema.
    content: Annotated[
        Optional[Dict[str, Any]],
        Field(description="The user-provided content, if any, as an object matching the requested schema."),
    ] = None


class ElicitationContentValue(RootModel[Union[str, int, float, bool, List[str]]]):
    root: Union[str, int, float, bool, List[str]]


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


class ElicitationBooleanPropertySchema(BooleanPropertySchema):
    type: Literal["boolean"]


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


class EnumOption(BaseModel):
    # The constant value for this option.
    const: Annotated[str, Field(description="The constant value for this option.")]
    # Human-readable title for this option.
    title: Annotated[str, Field(description="Human-readable title for this option.")]


class EnvVariable(BaseModel):
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
    # The name of the environment variable.
    name: Annotated[str, Field(description="The name of the environment variable.")]
    # The value to set for the environment variable.
    value: Annotated[str, Field(description="The value to set for the environment variable.")]


class FileSystemCapabilities(BaseModel):
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


class HttpHeader(BaseModel):
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
    # The name of the HTTP header.
    name: Annotated[str, Field(description="The name of the HTTP header.")]
    # The value to set for the HTTP header.
    value: Annotated[str, Field(description="The value to set for the HTTP header.")]


class Implementation(BaseModel):
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


class IntegerPropertySchema(BaseModel):
    # Default value.
    default: Annotated[Optional[int], Field(description="Default value.")] = None
    # Human-readable description.
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    # Maximum value (inclusive).
    maximum: Annotated[Optional[int], Field(description="Maximum value (inclusive).")] = None
    # Minimum value (inclusive).
    minimum: Annotated[Optional[int], Field(description="Minimum value (inclusive).")] = None
    # Optional title for the property.
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None


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


class ListSessionsRequest(BaseModel):
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
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Filter sessions by the exact ordered additional workspace roots. Each path must be absolute.
    #
    # This filter applies only when the field is present and non-empty. When
    # omitted or empty, no additional-root filter is applied.
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nFilter sessions by the exact ordered additional workspace roots. Each path must be absolute.\n\nThis filter applies only when the field is present and non-empty. When\nomitted or empty, no additional-root filter is applied.",
        ),
    ] = None
    # Opaque cursor token from a previous response's nextCursor field for cursor-based pagination
    cursor: Annotated[
        Optional[str],
        Field(
            description="Opaque cursor token from a previous response's nextCursor field for cursor-based pagination"
        ),
    ] = None
    # Filter sessions by working directory. Must be an absolute path.
    cwd: Annotated[
        Optional[str],
        Field(description="Filter sessions by working directory. Must be an absolute path."),
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


class McpCapabilities(BaseModel):
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
    # Agent supports [`McpServer::Http`].
    http: Annotated[Optional[bool], Field(description="Agent supports [`McpServer::Http`].")] = False
    # Agent supports [`McpServer::Sse`].
    sse: Annotated[Optional[bool], Field(description="Agent supports [`McpServer::Sse`].")] = False


class McpServerHttp(BaseModel):
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
    # HTTP headers to set when making requests to the MCP server.
    headers: Annotated[
        List[HttpHeader],
        Field(description="HTTP headers to set when making requests to the MCP server."),
    ]
    # Human-readable name identifying this MCP server.
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]
    # URL to the MCP server.
    url: Annotated[str, Field(description="URL to the MCP server.")]


class McpServerSse(BaseModel):
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
    # HTTP headers to set when making requests to the MCP server.
    headers: Annotated[
        List[HttpHeader],
        Field(description="HTTP headers to set when making requests to the MCP server."),
    ]
    # Human-readable name identifying this MCP server.
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]
    # URL to the MCP server.
    url: Annotated[str, Field(description="URL to the MCP server.")]


class McpServerStdio(BaseModel):
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
    # Command-line arguments to pass to the MCP server.
    args: Annotated[
        List[str],
        Field(description="Command-line arguments to pass to the MCP server."),
    ]
    # Path to the MCP server executable.
    command: Annotated[str, Field(description="Path to the MCP server executable.")]
    # Environment variables to set when launching the MCP server.
    env: Annotated[
        List[EnvVariable],
        Field(description="Environment variables to set when launching the MCP server."),
    ]
    # Human-readable name identifying this MCP server.
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]


class ModelInfo(BaseModel):
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
    # Optional description of the model.
    description: Annotated[Optional[str], Field(description="Optional description of the model.")] = None
    # Unique identifier for the model.
    model_id: Annotated[str, Field(alias="modelId", description="Unique identifier for the model.")]
    # Human-readable name of the model.
    name: Annotated[str, Field(description="Human-readable name of the model.")]


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


class NesEditHistoryCapabilities(BaseModel):
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
    # Maximum number of edit history entries the agent can use.
    max_count: Annotated[
        Optional[int],
        Field(
            alias="maxCount",
            description="Maximum number of edit history entries the agent can use.",
            ge=0,
        ),
    ] = None


class NesEditHistoryEntry(BaseModel):
    # A diff representing the edit.
    diff: Annotated[str, Field(description="A diff representing the edit.")]
    # The URI of the edited file.
    uri: Annotated[str, Field(description="The URI of the edited file.")]


class NesExcerpt(BaseModel):
    # The end line of the excerpt (zero-based).
    end_line: Annotated[
        int,
        Field(
            alias="endLine",
            description="The end line of the excerpt (zero-based).",
            ge=0,
        ),
    ]
    # The start line of the excerpt (zero-based).
    start_line: Annotated[
        int,
        Field(
            alias="startLine",
            description="The start line of the excerpt (zero-based).",
            ge=0,
        ),
    ]
    # The text content of the excerpt.
    text: Annotated[str, Field(description="The text content of the excerpt.")]


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


class NesRecentFile(BaseModel):
    # The language identifier.
    language_id: Annotated[str, Field(alias="languageId", description="The language identifier.")]
    # The full text content of the file.
    text: Annotated[str, Field(description="The full text content of the file.")]
    # The URI of the file.
    uri: Annotated[str, Field(description="The URI of the file.")]


class NesRecentFilesCapabilities(BaseModel):
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
    # Maximum number of recent files the agent can use.
    max_count: Annotated[
        Optional[int],
        Field(
            alias="maxCount",
            description="Maximum number of recent files the agent can use.",
            ge=0,
        ),
    ] = None


class NesRelatedSnippet(BaseModel):
    # The code excerpts.
    excerpts: Annotated[List[NesExcerpt], Field(description="The code excerpts.")]
    # The URI of the file containing the snippets.
    uri: Annotated[str, Field(description="The URI of the file containing the snippets.")]


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


class NesRepository(BaseModel):
    # The repository name.
    name: Annotated[str, Field(description="The repository name.")]
    # The repository owner.
    owner: Annotated[str, Field(description="The repository owner.")]
    # The remote URL of the repository.
    remote_url: Annotated[str, Field(alias="remoteUrl", description="The remote URL of the repository.")]


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


class NesSearchAndReplaceSuggestion(BaseModel):
    # Unique identifier for accept/reject tracking.
    id: Annotated[str, Field(description="Unique identifier for accept/reject tracking.")]
    # Whether `search` is a regular expression. Defaults to `false`.
    is_regex: Annotated[
        Optional[bool],
        Field(
            alias="isRegex",
            description="Whether `search` is a regular expression. Defaults to `false`.",
        ),
    ] = None
    # The replacement text.
    replace: Annotated[str, Field(description="The replacement text.")]
    # The text or pattern to find.
    search: Annotated[str, Field(description="The text or pattern to find.")]
    # The file URI to search within.
    uri: Annotated[str, Field(description="The file URI to search within.")]


class NesSearchAndReplaceSuggestionVariant(NesSearchAndReplaceSuggestion):
    kind: Literal["searchAndReplace"]


class NesUserActionsCapabilities(BaseModel):
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
    # Maximum number of user actions the agent can use.
    max_count: Annotated[
        Optional[int],
        Field(
            alias="maxCount",
            description="Maximum number of user actions the agent can use.",
            ge=0,
        ),
    ] = None


class NumberPropertySchema(BaseModel):
    # Default value.
    default: Annotated[Optional[float], Field(description="Default value.")] = None
    # Human-readable description.
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    # Maximum value (inclusive).
    maximum: Annotated[Optional[float], Field(description="Maximum value (inclusive).")] = None
    # Minimum value (inclusive).
    minimum: Annotated[Optional[float], Field(description="Minimum value (inclusive).")] = None
    # Optional title for the property.
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None


class Position(BaseModel):
    # Zero-based character offset (encoding-dependent).
    character: Annotated[
        int,
        Field(description="Zero-based character offset (encoding-dependent).", ge=0),
    ]
    # Zero-based line number.
    line: Annotated[int, Field(description="Zero-based line number.", ge=0)]


class PromptCapabilities(BaseModel):
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
    # Agent supports [`ContentBlock::Image`].
    image: Annotated[Optional[bool], Field(description="Agent supports [`ContentBlock::Image`].")] = False


class ProviderCurrentConfig(BaseModel):
    # Protocol currently used by this provider.
    api_type: Annotated[
        str,
        Field(alias="apiType", description="Protocol currently used by this provider."),
    ]
    # Base URL currently used by this provider.
    base_url: Annotated[
        str,
        Field(alias="baseUrl", description="Base URL currently used by this provider."),
    ]


class ProviderInfo(BaseModel):
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
    # Current effective non-secret routing config.
    # Null or omitted means provider is disabled.
    current: Annotated[
        Optional[ProviderCurrentConfig],
        Field(description="Current effective non-secret routing config.\nNull or omitted means provider is disabled."),
    ] = None
    # Provider identifier, for example "main" or "openai".
    id: Annotated[str, Field(description='Provider identifier, for example "main" or "openai".')]
    # Whether this provider is mandatory and cannot be disabled via `providers/disable`.
    # If true, clients must not call `providers/disable` for this id.
    required: Annotated[
        bool,
        Field(
            description="Whether this provider is mandatory and cannot be disabled via `providers/disable`.\nIf true, clients must not call `providers/disable` for this id."
        ),
    ]
    # Supported protocol types for this provider.
    supported: Annotated[List[str], Field(description="Supported protocol types for this provider.")]


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


class Range(BaseModel):
    # The end position (exclusive).
    end: Annotated[Position, Field(description="The end position (exclusive).")]
    # The start position (inclusive).
    start: Annotated[Position, Field(description="The start position (inclusive).")]


class ReadTextFileResponse(BaseModel):
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
    content: str


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


class DeniedOutcome(BaseModel):
    outcome: Literal["cancelled"]


class Role(Enum):
    assistant = "assistant"
    user = "user"


class SelectedPermissionOutcome(BaseModel):
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
    # The ID of the option the user selected.
    option_id: Annotated[
        str,
        Field(alias="optionId", description="The ID of the option the user selected."),
    ]


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


class SessionConfigBoolean(BaseModel):
    # The current value of the boolean option.
    current_value: Annotated[
        bool,
        Field(alias="currentValue", description="The current value of the boolean option."),
    ]


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


class SessionInfo(BaseModel):
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
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Authoritative ordered additional workspace roots for this session. Each path must be absolute.
    #
    # When omitted or empty, there are no additional roots for the session.
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nAuthoritative ordered additional workspace roots for this session. Each path must be absolute.\n\nWhen omitted or empty, there are no additional roots for the session.",
        ),
    ] = None
    # The working directory for this session. Must be an absolute path.
    cwd: Annotated[
        str,
        Field(description="The working directory for this session. Must be an absolute path."),
    ]
    # Unique identifier for the session
    session_id: Annotated[str, Field(alias="sessionId", description="Unique identifier for the session")]
    # Human-readable title for the session
    title: Annotated[Optional[str], Field(description="Human-readable title for the session")] = None
    # ISO 8601 timestamp of last activity
    updated_at: Annotated[
        Optional[str],
        Field(alias="updatedAt", description="ISO 8601 timestamp of last activity"),
    ] = None


class _SessionInfoUpdate(BaseModel):
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


class SessionModelState(BaseModel):
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
    # The set of models that the Agent can use
    available_models: Annotated[
        List[ModelInfo],
        Field(
            alias="availableModels",
            description="The set of models that the Agent can use",
        ),
    ]
    # The current model the Agent is in.
    current_model_id: Annotated[
        str,
        Field(alias="currentModelId", description="The current model the Agent is in."),
    ]


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


class SessionInfoUpdate(_SessionInfoUpdate):
    session_update: Annotated[Literal["session_info_update"], Field(alias="sessionUpdate")]


class SetProvidersRequest(BaseModel):
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
    # Protocol type for this provider.
    api_type: Annotated[str, Field(alias="apiType", description="Protocol type for this provider.")]
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
    # Provider id to configure.
    id: Annotated[str, Field(description="Provider id to configure.")]


class SetProvidersResponse(BaseModel):
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
    # The ID of the configuration option to set.
    config_id: Annotated[
        str,
        Field(alias="configId", description="The ID of the configuration option to set."),
    ]
    # The ID of the session to set the configuration option for.
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session to set the configuration option for.",
        ),
    ]
    type: Literal["boolean"]
    # The boolean value.
    value: Annotated[bool, Field(description="The boolean value.")]


class SetSessionConfigOptionSelectRequest(BaseModel):
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
    # The ID of the configuration option to set.
    config_id: Annotated[
        str,
        Field(alias="configId", description="The ID of the configuration option to set."),
    ]
    # The ID of the session to set the configuration option for.
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session to set the configuration option for.",
        ),
    ]
    # The value ID.
    value: Annotated[str, Field(description="The value ID.")]


class SetSessionModeRequest(BaseModel):
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
    # The ID of the mode to set.
    mode_id: Annotated[str, Field(alias="modeId", description="The ID of the mode to set.")]
    # The ID of the session to set the mode for.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The ID of the session to set the mode for."),
    ]


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


class SetSessionModelRequest(BaseModel):
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
    # The ID of the model to set.
    model_id: Annotated[str, Field(alias="modelId", description="The ID of the model to set.")]
    # The ID of the session to set the model for.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The ID of the session to set the model for."),
    ]


class SetSessionModelResponse(BaseModel):
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


class StartNesResponse(BaseModel):
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
    # The session ID for the newly started NES session.
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The session ID for the newly started NES session.",
        ),
    ]


class StringPropertySchema(BaseModel):
    # Default value.
    default: Annotated[Optional[str], Field(description="Default value.")] = None
    # Human-readable description.
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    # Enum values for untitled single-select enums.
    enum: Annotated[
        Optional[List[str]],
        Field(description="Enum values for untitled single-select enums."),
    ] = None
    # String format.
    format: Annotated[Optional[str], Field(description="String format.")] = None
    # Maximum string length.
    max_length: Annotated[
        Optional[int],
        Field(alias="maxLength", description="Maximum string length.", ge=0),
    ] = None
    # Minimum string length.
    min_length: Annotated[
        Optional[int],
        Field(alias="minLength", description="Minimum string length.", ge=0),
    ] = None
    # Titled enum options for titled single-select enums.
    one_of: Annotated[
        Optional[List[EnumOption]],
        Field(
            alias="oneOf",
            description="Titled enum options for titled single-select enums.",
        ),
    ] = None
    # Pattern the string must match.
    pattern: Annotated[Optional[str], Field(description="Pattern the string must match.")] = None
    # Optional title for the property.
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None


class Terminal(BaseModel):
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
    terminal_id: Annotated[str, Field(alias="terminalId")]


class TerminalExitStatus(BaseModel):
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


class TerminalOutputRequest(BaseModel):
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
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # The ID of the terminal to get output from.
    terminal_id: Annotated[
        str,
        Field(alias="terminalId", description="The ID of the terminal to get output from."),
    ]


class TerminalOutputResponse(BaseModel):
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
    # Exit status if the command has completed.
    exit_status: Annotated[
        Optional[TerminalExitStatus],
        Field(alias="exitStatus", description="Exit status if the command has completed."),
    ] = None
    # The terminal output captured so far.
    output: Annotated[str, Field(description="The terminal output captured so far.")]
    # Whether the output was truncated due to byte limits.
    truncated: Annotated[bool, Field(description="Whether the output was truncated due to byte limits.")]


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


class TextResourceContents(BaseModel):
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
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    text: str
    uri: str


class TitledMultiSelectItems(BaseModel):
    # Titled enum options.
    any_of: Annotated[List[EnumOption], Field(alias="anyOf", description="Titled enum options.")]


class FileEditToolCallContent(Diff):
    type: Literal["diff"]


class TerminalToolCallContent(Terminal):
    type: Literal["terminal"]


class ToolCallLocation(BaseModel):
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
    # Optional line number within the file.
    line: Annotated[Optional[int], Field(description="Optional line number within the file.", ge=0)] = None
    # The file path being accessed or modified.
    path: Annotated[str, Field(description="The file path being accessed or modified.")]


class UnstructuredCommandInput(BaseModel):
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
    # A hint to display when the input hasn't been provided yet
    hint: Annotated[
        str,
        Field(description="A hint to display when the input hasn't been provided yet"),
    ]


class UntitledMultiSelectItems(BaseModel):
    # Allowed enum values.
    enum: Annotated[List[str], Field(description="Allowed enum values.")]
    # Item type discriminator. Must be `"string"`.
    type: Annotated[str, Field(description='Item type discriminator. Must be `"string"`.')]


class Usage(BaseModel):
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
    # Sum of all token types across session.
    total_tokens: Annotated[
        int,
        Field(
            alias="totalTokens",
            description="Sum of all token types across session.",
            ge=0,
        ),
    ]


class _UsageUpdate(BaseModel):
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
    # Cumulative session cost (optional).
    cost: Annotated[Optional[Cost], Field(description="Cumulative session cost (optional).")] = None
    # Total context window size in tokens.
    size: Annotated[int, Field(description="Total context window size in tokens.", ge=0)]
    # Tokens currently in context.
    used: Annotated[int, Field(description="Tokens currently in context.", ge=0)]


class WaitForTerminalExitRequest(BaseModel):
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
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # The ID of the terminal to wait for.
    terminal_id: Annotated[
        str,
        Field(alias="terminalId", description="The ID of the terminal to wait for."),
    ]


class WaitForTerminalExitResponse(BaseModel):
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


class WorkspaceFolder(BaseModel):
    # The display name of the folder.
    name: Annotated[str, Field(description="The display name of the folder.")]
    # The URI of the folder.
    uri: Annotated[str, Field(description="The URI of the folder.")]


class WriteTextFileRequest(BaseModel):
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
    # The text content to write to the file.
    content: Annotated[str, Field(description="The text content to write to the file.")]
    # Absolute path to the file to write.
    path: Annotated[str, Field(description="Absolute path to the file to write.")]
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]


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


class AcceptNesNotification(BaseModel):
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
    # The ID of the accepted suggestion.
    id: Annotated[str, Field(description="The ID of the accepted suggestion.")]
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]


class AgentAuthCapabilities(BaseModel):
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
    # Whether the agent supports the logout method.
    #
    # By supplying `{}` it means that the agent supports the logout method.
    logout: Annotated[
        Optional[LogoutCapabilities],
        Field(
            description="Whether the agent supports the logout method.\n\nBy supplying `{}` it means that the agent supports the logout method."
        ),
    ] = None


class Annotations(BaseModel):
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
    audience: Optional[List[Role]] = None
    last_modified: Annotated[Optional[str], Field(alias="lastModified")] = None
    priority: Optional[float] = None


class AudioContent(BaseModel):
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
    annotations: Optional[Annotations] = None
    data: str
    mime_type: Annotated[str, Field(alias="mimeType")]


class EnvVarAuthMethod(AuthMethodEnvVar):
    type: Literal["env_var"]


class TerminalAuthMethod(AuthMethodTerminal):
    type: Literal["terminal"]


class AvailableCommandInput(RootModel[UnstructuredCommandInput]):
    # The input specification for a command.
    root: Annotated[
        UnstructuredCommandInput,
        Field(description="The input specification for a command."),
    ]


class CancelNotification(BaseModel):
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
    # The ID of the session to cancel operations for.
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session to cancel operations for.",
        ),
    ]


class CancelRequestNotification(BaseModel):
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
    # The ID of the request to cancel.
    request_id: Annotated[
        Optional[Union[int, str]],
        Field(alias="requestId", description="The ID of the request to cancel."),
    ] = None


class ClientNesCapabilities(BaseModel):
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


class CloseNesRequest(BaseModel):
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
    # The ID of the NES session to close.
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the NES session to close.")]


class CloseSessionRequest(BaseModel):
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
    # The ID of the session to close.
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to close.")]


class CompleteElicitationNotification(BaseModel):
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
    # The ID of the elicitation that completed.
    elicitation_id: Annotated[
        str,
        Field(
            alias="elicitationId",
            description="The ID of the elicitation that completed.",
        ),
    ]


class AudioContentBlock(AudioContent):
    type: Literal["audio"]


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


class CreateTerminalRequest(BaseModel):
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
    # Array of command arguments.
    args: Annotated[Optional[List[str]], Field(description="Array of command arguments.")] = None
    # The command to execute.
    command: Annotated[str, Field(description="The command to execute.")]
    # Working directory for the command (absolute path).
    cwd: Annotated[
        Optional[str],
        Field(description="Working directory for the command (absolute path)."),
    ] = None
    # Environment variables for the command.
    env: Annotated[
        Optional[List[EnvVariable]],
        Field(description="Environment variables for the command."),
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
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]


class _CurrentModeUpdate(BaseModel):
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
    # The ID of the current mode
    current_mode_id: Annotated[str, Field(alias="currentModeId", description="The ID of the current mode")]


class DidChangeDocumentNotification(BaseModel):
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
    # The content changes.
    content_changes: Annotated[
        List[TextDocumentContentChangeEvent],
        Field(alias="contentChanges", description="The content changes."),
    ]
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    # The URI of the changed document.
    uri: Annotated[str, Field(description="The URI of the changed document.")]
    # The new version number of the document.
    version: Annotated[int, Field(description="The new version number of the document.")]


class DidCloseDocumentNotification(BaseModel):
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
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    # The URI of the closed document.
    uri: Annotated[str, Field(description="The URI of the closed document.")]


class DidFocusDocumentNotification(BaseModel):
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
    # The current cursor position.
    position: Annotated[Position, Field(description="The current cursor position.")]
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    # The URI of the focused document.
    uri: Annotated[str, Field(description="The URI of the focused document.")]
    # The version number of the document.
    version: Annotated[int, Field(description="The version number of the document.")]
    # The portion of the file currently visible in the editor viewport.
    visible_range: Annotated[
        Range,
        Field(
            alias="visibleRange",
            description="The portion of the file currently visible in the editor viewport.",
        ),
    ]


class DidOpenDocumentNotification(BaseModel):
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
    # The language identifier of the document (e.g., "rust", "python").
    language_id: Annotated[
        str,
        Field(
            alias="languageId",
            description='The language identifier of the document (e.g., "rust", "python").',
        ),
    ]
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    # The full text content of the document.
    text: Annotated[str, Field(description="The full text content of the document.")]
    # The URI of the opened document.
    uri: Annotated[str, Field(description="The URI of the opened document.")]
    # The version number of the document.
    version: Annotated[int, Field(description="The version number of the document.")]


class DidSaveDocumentNotification(BaseModel):
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
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]
    # The URI of the saved document.
    uri: Annotated[str, Field(description="The URI of the saved document.")]


class ElicitationCapabilities(BaseModel):
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
    # Whether the client supports form-based elicitation.
    form: Annotated[
        Optional[ElicitationFormCapabilities],
        Field(description="Whether the client supports form-based elicitation."),
    ] = None
    # Whether the client supports URL-based elicitation.
    url: Annotated[
        Optional[ElicitationUrlCapabilities],
        Field(description="Whether the client supports URL-based elicitation."),
    ] = None


class ElicitationStringPropertySchema(StringPropertySchema):
    type: Literal["string"]


class ElicitationNumberPropertySchema(NumberPropertySchema):
    type: Literal["number"]


class ElicitationIntegerPropertySchema(IntegerPropertySchema):
    type: Literal["integer"]


class ElicitationRequestScope(BaseModel):
    # The request this elicitation is tied to.
    request_id: Annotated[
        Optional[Union[int, str]],
        Field(alias="requestId", description="The request this elicitation is tied to."),
    ] = None


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


class Error(BaseModel):
    # A number indicating the error type that occurred.
    # This must be an integer as defined in the JSON-RPC specification.
    code: Annotated[
        int,
        Field(
            description="A number indicating the error type that occurred.\nThis must be an integer as defined in the JSON-RPC specification."
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
    # A string providing a short description of the error.
    # The message should be limited to a concise single sentence.
    message: Annotated[
        str,
        Field(
            description="A string providing a short description of the error.\nThe message should be limited to a concise single sentence."
        ),
    ]


class ImageContent(BaseModel):
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
    annotations: Optional[Annotations] = None
    data: str
    mime_type: Annotated[str, Field(alias="mimeType")]
    uri: Optional[str] = None


class KillTerminalRequest(BaseModel):
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
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # The ID of the terminal to kill.
    terminal_id: Annotated[str, Field(alias="terminalId", description="The ID of the terminal to kill.")]


class ListProvidersResponse(BaseModel):
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
    # Configurable providers with current routing info suitable for UI display.
    providers: Annotated[
        List[ProviderInfo],
        Field(description="Configurable providers with current routing info suitable for UI display."),
    ]


class ListSessionsResponse(BaseModel):
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
    # Opaque cursor token. If present, pass this in the next request's cursor parameter
    # to fetch the next page. If absent, there are no more results.
    next_cursor: Annotated[
        Optional[str],
        Field(
            alias="nextCursor",
            description="Opaque cursor token. If present, pass this in the next request's cursor parameter\nto fetch the next page. If absent, there are no more results.",
        ),
    ] = None
    # Array of session information objects
    sessions: Annotated[List[SessionInfo], Field(description="Array of session information objects")]


class HttpMcpServer(McpServerHttp):
    type: Literal["http"]


class SseMcpServer(McpServerSse):
    type: Literal["sse"]


class MultiSelectPropertySchema(BaseModel):
    # Default selected values.
    default: Annotated[Optional[List[str]], Field(description="Default selected values.")] = None
    # Human-readable description.
    description: Annotated[Optional[str], Field(description="Human-readable description.")] = None
    # The items definition describing allowed values.
    items: Annotated[
        Union[UntitledMultiSelectItems, TitledMultiSelectItems],
        Field(description="The items definition describing allowed values."),
    ]
    # Maximum number of items to select.
    max_items: Annotated[
        Optional[int],
        Field(alias="maxItems", description="Maximum number of items to select.", ge=0),
    ] = None
    # Minimum number of items to select.
    min_items: Annotated[
        Optional[int],
        Field(alias="minItems", description="Minimum number of items to select.", ge=0),
    ] = None
    # Optional title for the property.
    title: Annotated[Optional[str], Field(description="Optional title for the property.")] = None


class NesContextCapabilities(BaseModel):
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
    # Whether the agent wants diagnostics context.
    diagnostics: Annotated[
        Optional[NesDiagnosticsCapabilities],
        Field(description="Whether the agent wants diagnostics context."),
    ] = None
    # Whether the agent wants edit history context.
    edit_history: Annotated[
        Optional[NesEditHistoryCapabilities],
        Field(
            alias="editHistory",
            description="Whether the agent wants edit history context.",
        ),
    ] = None
    # Whether the agent wants open files context.
    open_files: Annotated[
        Optional[NesOpenFilesCapabilities],
        Field(alias="openFiles", description="Whether the agent wants open files context."),
    ] = None
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
    # Whether the agent wants user actions context.
    user_actions: Annotated[
        Optional[NesUserActionsCapabilities],
        Field(
            alias="userActions",
            description="Whether the agent wants user actions context.",
        ),
    ] = None


class NesDiagnostic(BaseModel):
    # The diagnostic message.
    message: Annotated[str, Field(description="The diagnostic message.")]
    # The range of the diagnostic.
    range: Annotated[Range, Field(description="The range of the diagnostic.")]
    # The severity of the diagnostic.
    severity: Annotated[str, Field(description="The severity of the diagnostic.")]
    # The URI of the file containing the diagnostic.
    uri: Annotated[str, Field(description="The URI of the file containing the diagnostic.")]


class NesDocumentDidChangeCapabilities(BaseModel):
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
    # The sync kind the agent wants: `"full"` or `"incremental"`.
    sync_kind: Annotated[
        str,
        Field(
            alias="syncKind",
            description='The sync kind the agent wants: `"full"` or `"incremental"`.',
        ),
    ]


class NesDocumentEventCapabilities(BaseModel):
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
    # Whether the agent wants `document/didFocus` events.
    did_focus: Annotated[
        Optional[NesDocumentDidFocusCapabilities],
        Field(
            alias="didFocus",
            description="Whether the agent wants `document/didFocus` events.",
        ),
    ] = None
    # Whether the agent wants `document/didOpen` events.
    did_open: Annotated[
        Optional[NesDocumentDidOpenCapabilities],
        Field(
            alias="didOpen",
            description="Whether the agent wants `document/didOpen` events.",
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


class NesEventCapabilities(BaseModel):
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
    # Document event capabilities.
    document: Annotated[
        Optional[NesDocumentEventCapabilities],
        Field(description="Document event capabilities."),
    ] = None


class NesJumpSuggestion(BaseModel):
    # Unique identifier for accept/reject tracking.
    id: Annotated[str, Field(description="Unique identifier for accept/reject tracking.")]
    # The target position within the file.
    position: Annotated[Position, Field(description="The target position within the file.")]
    # The file to navigate to.
    uri: Annotated[str, Field(description="The file to navigate to.")]


class NesOpenFile(BaseModel):
    # The language identifier.
    language_id: Annotated[str, Field(alias="languageId", description="The language identifier.")]
    # Timestamp in milliseconds since epoch of when the file was last focused.
    last_focused_ms: Annotated[
        Optional[int],
        Field(
            alias="lastFocusedMs",
            description="Timestamp in milliseconds since epoch of when the file was last focused.",
            ge=0,
        ),
    ] = None
    # The URI of the file.
    uri: Annotated[str, Field(description="The URI of the file.")]
    # The visible range in the editor, if any.
    visible_range: Annotated[
        Optional[Range],
        Field(alias="visibleRange", description="The visible range in the editor, if any."),
    ] = None


class NesRenameSuggestion(BaseModel):
    # Unique identifier for accept/reject tracking.
    id: Annotated[str, Field(description="Unique identifier for accept/reject tracking.")]
    # The new name for the symbol.
    new_name: Annotated[str, Field(alias="newName", description="The new name for the symbol.")]
    # The position of the symbol to rename.
    position: Annotated[Position, Field(description="The position of the symbol to rename.")]
    # The file URI containing the symbol.
    uri: Annotated[str, Field(description="The file URI containing the symbol.")]


class NesJumpSuggestionVariant(NesJumpSuggestion):
    kind: Literal["jump"]


class NesRenameSuggestionVariant(NesRenameSuggestion):
    kind: Literal["rename"]


class NesTextEdit(BaseModel):
    # The replacement text.
    new_text: Annotated[str, Field(alias="newText", description="The replacement text.")]
    # The range to replace.
    range: Annotated[Range, Field(description="The range to replace.")]


class NesUserAction(BaseModel):
    # The kind of action (e.g., "insertChar", "cursorMovement").
    action: Annotated[
        str,
        Field(description='The kind of action (e.g., "insertChar", "cursorMovement").'),
    ]
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
    # The URI of the file where the action occurred.
    uri: Annotated[str, Field(description="The URI of the file where the action occurred.")]


class NewSessionRequest(BaseModel):
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
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Additional workspace roots for this session. Each path must be absolute.
    #
    # These expand the session's filesystem scope without changing `cwd`, which
    # remains the base for relative paths. When omitted or empty, no
    # additional roots are activated for the new session.
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nAdditional workspace roots for this session. Each path must be absolute.\n\nThese expand the session's filesystem scope without changing `cwd`, which\nremains the base for relative paths. When omitted or empty, no\nadditional roots are activated for the new session.",
        ),
    ] = None
    # The working directory for this session. Must be an absolute path.
    cwd: Annotated[
        str,
        Field(description="The working directory for this session. Must be an absolute path."),
    ]
    # List of MCP (Model Context Protocol) servers the agent should connect to.
    mcp_servers: Annotated[
        List[Union[HttpMcpServer, SseMcpServer, McpServerStdio]],
        Field(
            alias="mcpServers",
            description="List of MCP (Model Context Protocol) servers the agent should connect to.",
        ),
    ]


class PermissionOption(BaseModel):
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
    # Hint about the nature of this permission option.
    kind: Annotated[PermissionOptionKind, Field(description="Hint about the nature of this permission option.")]
    # Human-readable label to display to the user.
    name: Annotated[str, Field(description="Human-readable label to display to the user.")]
    # Unique identifier for this permission option.
    option_id: Annotated[
        str,
        Field(
            alias="optionId",
            description="Unique identifier for this permission option.",
        ),
    ]


class PlanEntry(BaseModel):
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


class PromptResponse(BaseModel):
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
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # The acknowledged user message ID.
    #
    # If the client provided a `messageId` in the [`PromptRequest`], the agent echoes it here
    # to confirm it was recorded. If the client did not provide one, the agent MAY assign one
    # and return it here. Absence of this field indicates the agent did not record a message ID.
    user_message_id: Annotated[
        Optional[str],
        Field(
            alias="userMessageId",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nThe acknowledged user message ID.\n\nIf the client provided a `messageId` in the [`PromptRequest`], the agent echoes it here\nto confirm it was recorded. If the client did not provide one, the agent MAY assign one\nand return it here. Absence of this field indicates the agent did not record a message ID.",
        ),
    ] = None


class ReadTextFileRequest(BaseModel):
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
    # Maximum number of lines to read.
    limit: Annotated[Optional[int], Field(description="Maximum number of lines to read.", ge=0)] = None
    # Line number to start reading from (1-based).
    line: Annotated[
        Optional[int],
        Field(description="Line number to start reading from (1-based).", ge=0),
    ] = None
    # Absolute path to the file to read.
    path: Annotated[str, Field(description="Absolute path to the file to read.")]
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]


class RejectNesNotification(BaseModel):
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
    # The ID of the rejected suggestion.
    id: Annotated[str, Field(description="The ID of the rejected suggestion.")]
    # The reason for rejection.
    reason: Annotated[Optional[str], Field(description="The reason for rejection.")] = None
    # The session ID for this notification.
    session_id: Annotated[
        str,
        Field(alias="sessionId", description="The session ID for this notification."),
    ]


class ReleaseTerminalRequest(BaseModel):
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
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # The ID of the terminal to release.
    terminal_id: Annotated[str, Field(alias="terminalId", description="The ID of the terminal to release.")]


class AllowedOutcome(SelectedPermissionOutcome):
    outcome: Literal["selected"]


class RequestPermissionResponse(BaseModel):
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
    # The user's decision on the permission request.
    outcome: Annotated[
        Union[DeniedOutcome, AllowedOutcome],
        Field(
            description="The user's decision on the permission request.",
            discriminator="outcome",
        ),
    ]


class ResourceLink(BaseModel):
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
    annotations: Optional[Annotations] = None
    description: Optional[str] = None
    mime_type: Annotated[Optional[str], Field(alias="mimeType")] = None
    name: str
    size: Optional[int] = None
    title: Optional[str] = None
    uri: str


class ResumeSessionRequest(BaseModel):
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
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Additional workspace roots to activate for this session. Each path must be absolute.
    #
    # When omitted or empty, no additional roots are activated. When non-empty,
    # this is the complete resulting additional-root list for the resumed
    # session.
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nAdditional workspace roots to activate for this session. Each path must be absolute.\n\nWhen omitted or empty, no additional roots are activated. When non-empty,\nthis is the complete resulting additional-root list for the resumed\nsession.",
        ),
    ] = None
    # The working directory for this session.
    cwd: Annotated[str, Field(description="The working directory for this session.")]
    # List of MCP servers to connect to for this session.
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, SseMcpServer, McpServerStdio]]],
        Field(
            alias="mcpServers",
            description="List of MCP servers to connect to for this session.",
        ),
    ] = None
    # The ID of the session to resume.
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to resume.")]


class SessionCapabilities(BaseModel):
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
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Whether the agent supports `additionalDirectories` on supported session lifecycle requests and `session/list`.
    additional_directories: Annotated[
        Optional[SessionAdditionalDirectoriesCapabilities],
        Field(
            alias="additionalDirectories",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nWhether the agent supports `additionalDirectories` on supported session lifecycle requests and `session/list`.",
        ),
    ] = None
    # Whether the agent supports `session/close`.
    close: Annotated[
        Optional[SessionCloseCapabilities],
        Field(description="Whether the agent supports `session/close`."),
    ] = None
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Whether the agent supports `session/fork`.
    fork: Annotated[
        Optional[SessionForkCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nWhether the agent supports `session/fork`."
        ),
    ] = None
    # Whether the agent supports `session/list`.
    list: Annotated[
        Optional[SessionListCapabilities],
        Field(description="Whether the agent supports `session/list`."),
    ] = None
    # Whether the agent supports `session/resume`.
    resume: Annotated[
        Optional[SessionResumeCapabilities],
        Field(description="Whether the agent supports `session/resume`."),
    ] = None


class SessionConfigOptionBoolean(SessionConfigBoolean):
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
    # Optional semantic category for this option (UX only).
    category: Annotated[
        Optional[str],
        Field(description="Optional semantic category for this option (UX only)."),
    ] = None
    # Optional description for the Client to display to the user.
    description: Annotated[
        Optional[str],
        Field(description="Optional description for the Client to display to the user."),
    ] = None
    # Unique identifier for the configuration option.
    id: Annotated[str, Field(description="Unique identifier for the configuration option.")]
    # Human-readable label for the option.
    name: Annotated[str, Field(description="Human-readable label for the option.")]
    type: Literal["boolean"]


class SessionConfigSelectOption(BaseModel):
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
    # Optional description for this option value.
    description: Annotated[Optional[str], Field(description="Optional description for this option value.")] = None
    # Human-readable label for this option value.
    name: Annotated[str, Field(description="Human-readable label for this option value.")]
    # Unique identifier for this option value.
    value: Annotated[str, Field(description="Unique identifier for this option value.")]


class SessionMode(BaseModel):
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
    description: Optional[str] = None
    # Unique identifier for a Session Mode.
    id: Annotated[str, Field(description="Unique identifier for a Session Mode.")]
    name: str


class SessionModeState(BaseModel):
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
    # The set of modes that the Agent can operate in
    available_modes: Annotated[
        List[SessionMode],
        Field(
            alias="availableModes",
            description="The set of modes that the Agent can operate in",
        ),
    ]
    # The current mode the Agent is in.
    current_mode_id: Annotated[
        str,
        Field(alias="currentModeId", description="The current mode the Agent is in."),
    ]


class CurrentModeUpdate(_CurrentModeUpdate):
    session_update: Annotated[Literal["current_mode_update"], Field(alias="sessionUpdate")]


class UsageUpdate(_UsageUpdate):
    session_update: Annotated[Literal["usage_update"], Field(alias="sessionUpdate")]


class StartNesRequest(BaseModel):
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
    # Repository metadata, if the workspace is a git repository.
    repository: Annotated[
        Optional[NesRepository],
        Field(description="Repository metadata, if the workspace is a git repository."),
    ] = None
    # The workspace folders.
    workspace_folders: Annotated[
        Optional[List[WorkspaceFolder]],
        Field(alias="workspaceFolders", description="The workspace folders."),
    ] = None
    # The root URI of the workspace.
    workspace_uri: Annotated[
        Optional[str],
        Field(alias="workspaceUri", description="The root URI of the workspace."),
    ] = None


class TextContent(BaseModel):
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
    annotations: Optional[Annotations] = None
    text: str


class AgentErrorMessage(BaseModel):
    error: Error
    # JSON RPC Request Id
    #
    # An identifier established by the Client that MUST contain a String, Number, or NULL value if included. If it is not included it is assumed to be a notification. The value SHOULD normally not be Null [1] and Numbers SHOULD NOT contain fractional parts [2]
    #
    # The Server MUST reply with the same value in the Response object if included. This member is used to correlate the context between the two objects.
    #
    # [1] The use of Null as a value for the id member in a Request object is discouraged, because this specification uses a value of Null for Responses with an unknown id. Also, because JSON-RPC 1.0 uses an id value of Null for Notifications this could cause confusion in handling.
    #
    # [2] Fractional parts may be problematic, since many decimal fractions cannot be represented exactly as binary fractions.
    id: Annotated[
        Optional[Union[int, str]],
        Field(
            description="JSON RPC Request Id\n\nAn identifier established by the Client that MUST contain a String, Number, or NULL value if included. If it is not included it is assumed to be a notification. The value SHOULD normally not be Null [1] and Numbers SHOULD NOT contain fractional parts [2]\n\nThe Server MUST reply with the same value in the Response object if included. This member is used to correlate the context between the two objects.\n\n[1] The use of Null as a value for the id member in a Request object is discouraged, because this specification uses a value of Null for Responses with an unknown id. Also, because JSON-RPC 1.0 uses an id value of Null for Notifications this could cause confusion in handling.\n\n[2] Fractional parts may be problematic, since many decimal fractions cannot be represented exactly as binary fractions."
        ),
    ] = None


class AvailableCommand(BaseModel):
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
    # Human-readable description of what the command does.
    description: Annotated[str, Field(description="Human-readable description of what the command does.")]
    # Input for the command if required
    input: Annotated[
        Optional[AvailableCommandInput],
        Field(description="Input for the command if required"),
    ] = None
    # Command name (e.g., `create_plan`, `research_codebase`).
    name: Annotated[
        str,
        Field(description="Command name (e.g., `create_plan`, `research_codebase`)."),
    ]


class _AvailableCommandsUpdate(BaseModel):
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
    # Commands the agent can execute
    available_commands: Annotated[
        List[AvailableCommand],
        Field(alias="availableCommands", description="Commands the agent can execute"),
    ]


class ClientCapabilities(BaseModel):
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
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nAuthentication capabilities supported by the client.\nDetermines which authentication method types the agent may include\nin its `InitializeResponse`."
        ),
    ] = {"terminal": False}
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Elicitation capabilities supported by the client.
    # Determines which elicitation modes the agent may use.
    elicitation: Annotated[
        Optional[ElicitationCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nElicitation capabilities supported by the client.\nDetermines which elicitation modes the agent may use."
        ),
    ] = None
    # File system capabilities supported by the client.
    # Determines which file operations the agent can request.
    fs: Annotated[
        Optional[FileSystemCapabilities],
        Field(
            description="File system capabilities supported by the client.\nDetermines which file operations the agent can request."
        ),
    ] = FileSystemCapabilities()
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # NES (Next Edit Suggestions) capabilities supported by the client.
    nes: Annotated[
        Optional[ClientNesCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nNES (Next Edit Suggestions) capabilities supported by the client."
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
    # Whether the Client support all `terminal/*` methods.
    terminal: Annotated[
        Optional[bool],
        Field(description="Whether the Client support all `terminal/*` methods."),
    ] = False


class ClientNotification(BaseModel):
    method: str
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
            Any,
        ]
    ] = None


class ClientResponseMessage(BaseModel):
    # JSON RPC Request Id
    #
    # An identifier established by the Client that MUST contain a String, Number, or NULL value if included. If it is not included it is assumed to be a notification. The value SHOULD normally not be Null [1] and Numbers SHOULD NOT contain fractional parts [2]
    #
    # The Server MUST reply with the same value in the Response object if included. This member is used to correlate the context between the two objects.
    #
    # [1] The use of Null as a value for the id member in a Request object is discouraged, because this specification uses a value of Null for Responses with an unknown id. Also, because JSON-RPC 1.0 uses an id value of Null for Notifications this could cause confusion in handling.
    #
    # [2] Fractional parts may be problematic, since many decimal fractions cannot be represented exactly as binary fractions.
    id: Annotated[
        Optional[Union[int, str]],
        Field(
            description="JSON RPC Request Id\n\nAn identifier established by the Client that MUST contain a String, Number, or NULL value if included. If it is not included it is assumed to be a notification. The value SHOULD normally not be Null [1] and Numbers SHOULD NOT contain fractional parts [2]\n\nThe Server MUST reply with the same value in the Response object if included. This member is used to correlate the context between the two objects.\n\n[1] The use of Null as a value for the id member in a Request object is discouraged, because this specification uses a value of Null for Responses with an unknown id. Also, because JSON-RPC 1.0 uses an id value of Null for Notifications this could cause confusion in handling.\n\n[2] Fractional parts may be problematic, since many decimal fractions cannot be represented exactly as binary fractions."
        ),
    ] = None
    # All possible responses that a client can send to an agent.
    #
    # This enum is used internally for routing RPC responses. You typically won't need
    # to use this directly - the responses are handled automatically by the connection.
    #
    # These are responses to the corresponding `AgentRequest` variants.
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
            Union[
                AcceptElicitationResponse,
                DeclineElicitationResponse,
                CancelElicitationResponse,
            ],
            Any,
        ],
        Field(
            description="All possible responses that a client can send to an agent.\n\nThis enum is used internally for routing RPC responses. You typically won't need\nto use this directly - the responses are handled automatically by the connection.\n\nThese are responses to the corresponding `AgentRequest` variants."
        ),
    ]


class ClientErrorMessage(BaseModel):
    error: Error
    # JSON RPC Request Id
    #
    # An identifier established by the Client that MUST contain a String, Number, or NULL value if included. If it is not included it is assumed to be a notification. The value SHOULD normally not be Null [1] and Numbers SHOULD NOT contain fractional parts [2]
    #
    # The Server MUST reply with the same value in the Response object if included. This member is used to correlate the context between the two objects.
    #
    # [1] The use of Null as a value for the id member in a Request object is discouraged, because this specification uses a value of Null for Responses with an unknown id. Also, because JSON-RPC 1.0 uses an id value of Null for Notifications this could cause confusion in handling.
    #
    # [2] Fractional parts may be problematic, since many decimal fractions cannot be represented exactly as binary fractions.
    id: Annotated[
        Optional[Union[int, str]],
        Field(
            description="JSON RPC Request Id\n\nAn identifier established by the Client that MUST contain a String, Number, or NULL value if included. If it is not included it is assumed to be a notification. The value SHOULD normally not be Null [1] and Numbers SHOULD NOT contain fractional parts [2]\n\nThe Server MUST reply with the same value in the Response object if included. This member is used to correlate the context between the two objects.\n\n[1] The use of Null as a value for the id member in a Request object is discouraged, because this specification uses a value of Null for Responses with an unknown id. Also, because JSON-RPC 1.0 uses an id value of Null for Notifications this could cause confusion in handling.\n\n[2] Fractional parts may be problematic, since many decimal fractions cannot be represented exactly as binary fractions."
        ),
    ] = None


class ClientResponse(RootModel[Union[ClientResponseMessage, ClientErrorMessage]]):
    root: Union[ClientResponseMessage, ClientErrorMessage]


class TextContentBlock(TextContent):
    type: Literal["text"]


class ImageContentBlock(ImageContent):
    type: Literal["image"]


class ResourceContentBlock(ResourceLink):
    type: Literal["resource_link"]


class CreateUrlElicitationRequest(BaseModel):
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
    # A human-readable message describing what input is needed.
    message: Annotated[
        str,
        Field(description="A human-readable message describing what input is needed."),
    ]
    mode: Literal["url"]


class ElicitationMultiSelectPropertySchema(MultiSelectPropertySchema):
    type: Literal["array"]


class ElicitationSchema(BaseModel):
    # Optional description of what this schema represents.
    description: Annotated[
        Optional[str],
        Field(description="Optional description of what this schema represents."),
    ] = None
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
                ],
            ]
        ],
        Field(description="Property definitions (must be primitive types)."),
    ] = {}
    # List of required property names.
    required: Annotated[Optional[List[str]], Field(description="List of required property names.")] = None
    # Optional title for the schema.
    title: Annotated[Optional[str], Field(description="Optional title for the schema.")] = None
    # Type discriminator. Always `"object"`.
    type: Annotated[Optional[str], Field(description='Type discriminator. Always `"object"`.')] = "object"


class EmbeddedResource(BaseModel):
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
    annotations: Optional[Annotations] = None
    # Resource content that can be embedded in a message.
    resource: Annotated[
        Union[TextResourceContents, BlobResourceContents],
        Field(description="Resource content that can be embedded in a message."),
    ]


class ForkSessionRequest(BaseModel):
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
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Additional workspace roots to activate for this session. Each path must be absolute.
    #
    # When omitted or empty, no additional roots are activated. When non-empty,
    # this is the complete resulting additional-root list for the forked
    # session.
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nAdditional workspace roots to activate for this session. Each path must be absolute.\n\nWhen omitted or empty, no additional roots are activated. When non-empty,\nthis is the complete resulting additional-root list for the forked\nsession.",
        ),
    ] = None
    # The working directory for this session.
    cwd: Annotated[str, Field(description="The working directory for this session.")]
    # List of MCP servers to connect to for this session.
    mcp_servers: Annotated[
        Optional[List[Union[HttpMcpServer, SseMcpServer, McpServerStdio]]],
        Field(
            alias="mcpServers",
            description="List of MCP servers to connect to for this session.",
        ),
    ] = None
    # The ID of the session to fork.
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to fork.")]


class InitializeRequest(BaseModel):
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
    # Capabilities supported by the client.
    client_capabilities: Annotated[
        Optional[ClientCapabilities],
        Field(
            alias="clientCapabilities",
            description="Capabilities supported by the client.",
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


class LoadSessionRequest(BaseModel):
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
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Additional workspace roots to activate for this session. Each path must be absolute.
    #
    # When omitted or empty, no additional roots are activated. When non-empty,
    # this is the complete resulting additional-root list for the loaded
    # session.
    additional_directories: Annotated[
        Optional[List[str]],
        Field(
            alias="additionalDirectories",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nAdditional workspace roots to activate for this session. Each path must be absolute.\n\nWhen omitted or empty, no additional roots are activated. When non-empty,\nthis is the complete resulting additional-root list for the loaded\nsession.",
        ),
    ] = None
    # The working directory for this session.
    cwd: Annotated[str, Field(description="The working directory for this session.")]
    # List of MCP servers to connect to for this session.
    mcp_servers: Annotated[
        List[Union[HttpMcpServer, SseMcpServer, McpServerStdio]],
        Field(
            alias="mcpServers",
            description="List of MCP servers to connect to for this session.",
        ),
    ]
    # The ID of the session to load.
    session_id: Annotated[str, Field(alias="sessionId", description="The ID of the session to load.")]


class NesCapabilities(BaseModel):
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
    # Context the agent wants attached to each suggestion request.
    context: Annotated[
        Optional[NesContextCapabilities],
        Field(description="Context the agent wants attached to each suggestion request."),
    ] = None
    # Events the agent wants to receive.
    events: Annotated[
        Optional[NesEventCapabilities],
        Field(description="Events the agent wants to receive."),
    ] = None


class NesEditSuggestion(BaseModel):
    # Optional suggested cursor position after applying edits.
    cursor_position: Annotated[
        Optional[Position],
        Field(
            alias="cursorPosition",
            description="Optional suggested cursor position after applying edits.",
        ),
    ] = None
    # The text edits to apply.
    edits: Annotated[List[NesTextEdit], Field(description="The text edits to apply.")]
    # Unique identifier for accept/reject tracking.
    id: Annotated[str, Field(description="Unique identifier for accept/reject tracking.")]
    # The URI of the file to edit.
    uri: Annotated[str, Field(description="The URI of the file to edit.")]


class NesSuggestContext(BaseModel):
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
    # Current diagnostics (errors, warnings).
    diagnostics: Annotated[
        Optional[List[NesDiagnostic]],
        Field(description="Current diagnostics (errors, warnings)."),
    ] = None
    # Recent edit history.
    edit_history: Annotated[
        Optional[List[NesEditHistoryEntry]],
        Field(alias="editHistory", description="Recent edit history."),
    ] = None
    # Currently open files in the editor.
    open_files: Annotated[
        Optional[List[NesOpenFile]],
        Field(alias="openFiles", description="Currently open files in the editor."),
    ] = None
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
    # Recent user actions (typing, navigation, etc.).
    user_actions: Annotated[
        Optional[List[NesUserAction]],
        Field(
            alias="userActions",
            description="Recent user actions (typing, navigation, etc.).",
        ),
    ] = None


class NesEditSuggestionVariant(NesEditSuggestion):
    kind: Literal["edit"]


class Plan(BaseModel):
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


class SessionConfigSelectGroup(BaseModel):
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
    # Unique identifier for this group.
    group: Annotated[str, Field(description="Unique identifier for this group.")]
    # Human-readable label for this group.
    name: Annotated[str, Field(description="Human-readable label for this group.")]
    # The set of option values in this group.
    options: Annotated[
        List[SessionConfigSelectOption],
        Field(description="The set of option values in this group."),
    ]


class AgentPlanUpdate(Plan):
    session_update: Annotated[Literal["plan"], Field(alias="sessionUpdate")]


class AvailableCommandsUpdate(_AvailableCommandsUpdate):
    session_update: Annotated[Literal["available_commands_update"], Field(alias="sessionUpdate")]


class SuggestNesRequest(BaseModel):
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
    # Context for the suggestion, included based on agent capabilities.
    context: Annotated[
        Optional[NesSuggestContext],
        Field(description="Context for the suggestion, included based on agent capabilities."),
    ] = None
    # The current cursor position.
    position: Annotated[Position, Field(description="The current cursor position.")]
    # The current text selection range, if any.
    selection: Annotated[Optional[Range], Field(description="The current text selection range, if any.")] = None
    # The session ID for this request.
    session_id: Annotated[str, Field(alias="sessionId", description="The session ID for this request.")]
    # What triggered this suggestion request.
    trigger_kind: Annotated[
        str,
        Field(alias="triggerKind", description="What triggered this suggestion request."),
    ]
    # The URI of the document to suggest for.
    uri: Annotated[str, Field(description="The URI of the document to suggest for.")]
    # The version number of the document.
    version: Annotated[int, Field(description="The version number of the document.")]


class SuggestNesResponse(BaseModel):
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
    # The list of suggestions.
    suggestions: Annotated[
        List[
            Union[
                NesEditSuggestionVariant,
                NesJumpSuggestionVariant,
                NesRenameSuggestionVariant,
                NesSearchAndReplaceSuggestionVariant,
            ]
        ],
        Field(description="The list of suggestions."),
    ]


class AgentCapabilities(BaseModel):
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
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Authentication-related capabilities supported by the agent.
    auth: Annotated[
        Optional[AgentAuthCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nAuthentication-related capabilities supported by the agent."
        ),
    ] = {}
    # Whether the agent supports `session/load`.
    load_session: Annotated[
        Optional[bool],
        Field(
            alias="loadSession",
            description="Whether the agent supports `session/load`.",
        ),
    ] = False
    # MCP capabilities supported by the agent.
    mcp_capabilities: Annotated[
        Optional[McpCapabilities],
        Field(
            alias="mcpCapabilities",
            description="MCP capabilities supported by the agent.",
        ),
    ] = McpCapabilities()
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # NES (Next Edit Suggestions) capabilities supported by the agent.
    nes: Annotated[
        Optional[NesCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nNES (Next Edit Suggestions) capabilities supported by the agent."
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
    # Prompt capabilities supported by the agent.
    prompt_capabilities: Annotated[
        Optional[PromptCapabilities],
        Field(
            alias="promptCapabilities",
            description="Prompt capabilities supported by the agent.",
        ),
    ] = PromptCapabilities()
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Provider configuration capabilities supported by the agent.
    #
    # By supplying `{}` it means that the agent supports provider configuration methods.
    providers: Annotated[
        Optional[ProvidersCapabilities],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nProvider configuration capabilities supported by the agent.\n\nBy supplying `{}` it means that the agent supports provider configuration methods."
        ),
    ] = None
    session_capabilities: Annotated[Optional[SessionCapabilities], Field(alias="sessionCapabilities")] = (
        SessionCapabilities()
    )


class EmbeddedResourceContentBlock(EmbeddedResource):
    type: Literal["resource"]


class ContentChunk(BaseModel):
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
    # A single item of content
    content: Annotated[
        Union[
            TextContentBlock, ImageContentBlock, AudioContentBlock, ResourceContentBlock, EmbeddedResourceContentBlock
        ],
        Field(description="A single item of content", discriminator="type"),
    ]
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # A unique identifier for the message this chunk belongs to.
    #
    # All chunks belonging to the same message share the same `messageId`.
    # A change in `messageId` indicates a new message has started.
    # Both clients and agents MUST use UUID format for message IDs.
    message_id: Annotated[
        Optional[str],
        Field(
            alias="messageId",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nA unique identifier for the message this chunk belongs to.\n\nAll chunks belonging to the same message share the same `messageId`.\nA change in `messageId` indicates a new message has started.\nBoth clients and agents MUST use UUID format for message IDs.",
        ),
    ] = None


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


class InitializeResponse(BaseModel):
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
    # Capabilities supported by the agent.
    agent_capabilities: Annotated[
        Optional[AgentCapabilities],
        Field(
            alias="agentCapabilities",
            description="Capabilities supported by the agent.",
        ),
    ] = AgentCapabilities()
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
    # Authentication methods supported by the agent.
    auth_methods: Annotated[
        Optional[List[Union[EnvVarAuthMethod, TerminalAuthMethod, AuthMethodAgent]]],
        Field(
            alias="authMethods",
            description="Authentication methods supported by the agent.",
        ),
    ] = []
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


class PromptRequest(BaseModel):
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
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # A client-generated unique identifier for this user message.
    #
    # If provided, the Agent SHOULD echo this value as `userMessageId` in the
    # [`PromptResponse`] to confirm it was recorded.
    # Both clients and agents MUST use UUID format for message IDs.
    message_id: Annotated[
        Optional[str],
        Field(
            alias="messageId",
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nA client-generated unique identifier for this user message.\n\nIf provided, the Agent SHOULD echo this value as `userMessageId` in the\n[`PromptResponse`] to confirm it was recorded.\nBoth clients and agents MUST use UUID format for message IDs.",
        ),
    ] = None
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
            Union[
                TextContentBlock,
                ImageContentBlock,
                AudioContentBlock,
                ResourceContentBlock,
                EmbeddedResourceContentBlock,
            ]
        ],
        Field(
            description="The blocks of content that compose the user's message.\n\nAs a baseline, the Agent MUST support [`ContentBlock::Text`] and [`ContentBlock::ResourceLink`],\nwhile other variants are optionally enabled via [`PromptCapabilities`].\n\nThe Client MUST adapt its interface according to [`PromptCapabilities`].\n\nThe client MAY include referenced pieces of context as either\n[`ContentBlock::Resource`] or [`ContentBlock::ResourceLink`].\n\nWhen available, [`ContentBlock::Resource`] is preferred\nas it avoids extra round-trips and allows the message to include\npieces of context from sources the agent may not have access to."
        ),
    ]
    # The ID of the session to send this user message to
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="The ID of the session to send this user message to",
        ),
    ]


class SessionConfigSelect(BaseModel):
    # The currently selected value.
    current_value: Annotated[str, Field(alias="currentValue", description="The currently selected value.")]
    # The set of selectable options.
    options: Annotated[
        Union[List[SessionConfigSelectOption], List[SessionConfigSelectGroup]],
        Field(description="The set of selectable options."),
    ]


class UserMessageChunk(ContentChunk):
    session_update: Annotated[Literal["user_message_chunk"], Field(alias="sessionUpdate")]


class AgentMessageChunk(ContentChunk):
    session_update: Annotated[Literal["agent_message_chunk"], Field(alias="sessionUpdate")]


class AgentThoughtChunk(ContentChunk):
    session_update: Annotated[Literal["agent_thought_chunk"], Field(alias="sessionUpdate")]


class ClientRequest(BaseModel):
    # JSON RPC Request Id
    #
    # An identifier established by the Client that MUST contain a String, Number, or NULL value if included. If it is not included it is assumed to be a notification. The value SHOULD normally not be Null [1] and Numbers SHOULD NOT contain fractional parts [2]
    #
    # The Server MUST reply with the same value in the Response object if included. This member is used to correlate the context between the two objects.
    #
    # [1] The use of Null as a value for the id member in a Request object is discouraged, because this specification uses a value of Null for Responses with an unknown id. Also, because JSON-RPC 1.0 uses an id value of Null for Notifications this could cause confusion in handling.
    #
    # [2] Fractional parts may be problematic, since many decimal fractions cannot be represented exactly as binary fractions.
    id: Annotated[
        Optional[Union[int, str]],
        Field(
            description="JSON RPC Request Id\n\nAn identifier established by the Client that MUST contain a String, Number, or NULL value if included. If it is not included it is assumed to be a notification. The value SHOULD normally not be Null [1] and Numbers SHOULD NOT contain fractional parts [2]\n\nThe Server MUST reply with the same value in the Response object if included. This member is used to correlate the context between the two objects.\n\n[1] The use of Null as a value for the id member in a Request object is discouraged, because this specification uses a value of Null for Responses with an unknown id. Also, because JSON-RPC 1.0 uses an id value of Null for Notifications this could cause confusion in handling.\n\n[2] Fractional parts may be problematic, since many decimal fractions cannot be represented exactly as binary fractions."
        ),
    ] = None
    method: str
    params: Optional[
        Union[
            InitializeRequest,
            AuthenticateRequest,
            ListProvidersRequest,
            SetProvidersRequest,
            DisableProvidersRequest,
            LogoutRequest,
            NewSessionRequest,
            LoadSessionRequest,
            ListSessionsRequest,
            ForkSessionRequest,
            ResumeSessionRequest,
            CloseSessionRequest,
            SetSessionModeRequest,
            PromptRequest,
            SetSessionModelRequest,
            StartNesRequest,
            SuggestNesRequest,
            CloseNesRequest,
            Union[SetSessionConfigOptionBooleanRequest, SetSessionConfigOptionSelectRequest],
            Any,
        ]
    ] = None


class Content(BaseModel):
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
    # The actual content block.
    content: Annotated[
        Union[
            TextContentBlock, ImageContentBlock, AudioContentBlock, ResourceContentBlock, EmbeddedResourceContentBlock
        ],
        Field(description="The actual content block.", discriminator="type"),
    ]


class CreateFormElicitationRequest(BaseModel):
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
    # A human-readable message describing what input is needed.
    message: Annotated[
        str,
        Field(description="A human-readable message describing what input is needed."),
    ]
    mode: Literal["form"]


class SessionConfigOptionSelect(SessionConfigSelect):
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
    # Optional semantic category for this option (UX only).
    category: Annotated[
        Optional[str],
        Field(description="Optional semantic category for this option (UX only)."),
    ] = None
    # Optional description for the Client to display to the user.
    description: Annotated[
        Optional[str],
        Field(description="Optional description for the Client to display to the user."),
    ] = None
    # Unique identifier for the configuration option.
    id: Annotated[str, Field(description="Unique identifier for the configuration option.")]
    # Human-readable label for the option.
    name: Annotated[str, Field(description="Human-readable label for the option.")]
    type: Literal["select"]


class SetSessionConfigOptionResponse(BaseModel):
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
    # The full set of configuration options and their current values.
    config_options: Annotated[
        List[Union[SessionConfigOptionSelect, SessionConfigOptionBoolean]],
        Field(
            alias="configOptions",
            description="The full set of configuration options and their current values.",
        ),
    ]


class ContentToolCallContent(Content):
    type: Literal["content"]


class ToolCallUpdate(BaseModel):
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
    # Replace the content collection.
    content: Annotated[
        Optional[List[Union[ContentToolCallContent, FileEditToolCallContent, TerminalToolCallContent]]],
        Field(description="Replace the content collection."),
    ] = None
    # Update the tool kind.
    kind: Annotated[Optional[ToolKind], Field(description="Update the tool kind.")] = None
    # Replace the locations collection.
    locations: Annotated[
        Optional[List[ToolCallLocation]],
        Field(description="Replace the locations collection."),
    ] = None
    # Update the raw input.
    raw_input: Annotated[Optional[Any], Field(alias="rawInput", description="Update the raw input.")] = None
    # Update the raw output.
    raw_output: Annotated[Optional[Any], Field(alias="rawOutput", description="Update the raw output.")] = None
    # Update the execution status.
    status: Annotated[Optional[ToolCallStatus], Field(description="Update the execution status.")] = None
    # Update the human-readable title.
    title: Annotated[Optional[str], Field(description="Update the human-readable title.")] = None
    # The ID of the tool call being updated.
    tool_call_id: Annotated[
        str,
        Field(alias="toolCallId", description="The ID of the tool call being updated."),
    ]


class _ConfigOptionUpdate(BaseModel):
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
    # The full set of configuration options and their current values.
    config_options: Annotated[
        List[Union[SessionConfigOptionSelect, SessionConfigOptionBoolean]],
        Field(
            alias="configOptions",
            description="The full set of configuration options and their current values.",
        ),
    ]


class ForkSessionResponse(BaseModel):
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
    # Initial session configuration options if supported by the Agent.
    config_options: Annotated[
        Optional[List[Union[SessionConfigOptionSelect, SessionConfigOptionBoolean]]],
        Field(
            alias="configOptions",
            description="Initial session configuration options if supported by the Agent.",
        ),
    ] = None
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Initial model state if supported by the Agent
    models: Annotated[
        Optional[SessionModelState],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nInitial model state if supported by the Agent"
        ),
    ] = None
    # Initial mode state if supported by the Agent
    #
    # See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    modes: Annotated[
        Optional[SessionModeState],
        Field(
            description="Initial mode state if supported by the Agent\n\nSee protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)"
        ),
    ] = None
    # Unique identifier for the newly created forked session.
    session_id: Annotated[
        str,
        Field(
            alias="sessionId",
            description="Unique identifier for the newly created forked session.",
        ),
    ]


class LoadSessionResponse(BaseModel):
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
    # Initial session configuration options if supported by the Agent.
    config_options: Annotated[
        Optional[List[Union[SessionConfigOptionSelect, SessionConfigOptionBoolean]]],
        Field(
            alias="configOptions",
            description="Initial session configuration options if supported by the Agent.",
        ),
    ] = None
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Initial model state if supported by the Agent
    models: Annotated[
        Optional[SessionModelState],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nInitial model state if supported by the Agent"
        ),
    ] = None
    # Initial mode state if supported by the Agent
    #
    # See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    modes: Annotated[
        Optional[SessionModeState],
        Field(
            description="Initial mode state if supported by the Agent\n\nSee protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)"
        ),
    ] = None


class NewSessionResponse(BaseModel):
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
    # Initial session configuration options if supported by the Agent.
    config_options: Annotated[
        Optional[List[Union[SessionConfigOptionSelect, SessionConfigOptionBoolean]]],
        Field(
            alias="configOptions",
            description="Initial session configuration options if supported by the Agent.",
        ),
    ] = None
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Initial model state if supported by the Agent
    models: Annotated[
        Optional[SessionModelState],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nInitial model state if supported by the Agent"
        ),
    ] = None
    # Initial mode state if supported by the Agent
    #
    # See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    modes: Annotated[
        Optional[SessionModeState],
        Field(
            description="Initial mode state if supported by the Agent\n\nSee protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)"
        ),
    ] = None
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


class RequestPermissionRequest(BaseModel):
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
    # Available permission options for the user to choose from.
    options: Annotated[
        List[PermissionOption],
        Field(description="Available permission options for the user to choose from."),
    ]
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


class ResumeSessionResponse(BaseModel):
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
    # Initial session configuration options if supported by the Agent.
    config_options: Annotated[
        Optional[List[Union[SessionConfigOptionSelect, SessionConfigOptionBoolean]]],
        Field(
            alias="configOptions",
            description="Initial session configuration options if supported by the Agent.",
        ),
    ] = None
    # **UNSTABLE**
    #
    # This capability is not part of the spec yet, and may be removed or changed at any point.
    #
    # Initial model state if supported by the Agent
    models: Annotated[
        Optional[SessionModelState],
        Field(
            description="**UNSTABLE**\n\nThis capability is not part of the spec yet, and may be removed or changed at any point.\n\nInitial model state if supported by the Agent"
        ),
    ] = None
    # Initial mode state if supported by the Agent
    #
    # See protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)
    modes: Annotated[
        Optional[SessionModeState],
        Field(
            description="Initial mode state if supported by the Agent\n\nSee protocol docs: [Session Modes](https://agentclientprotocol.com/protocol/session-modes)"
        ),
    ] = None


class ToolCallProgress(ToolCallUpdate):
    session_update: Annotated[Literal["tool_call_update"], Field(alias="sessionUpdate")]


class ConfigOptionUpdate(_ConfigOptionUpdate):
    session_update: Annotated[Literal["config_option_update"], Field(alias="sessionUpdate")]


class ToolCall(BaseModel):
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
    # Content produced by the tool call.
    content: Annotated[
        Optional[List[Union[ContentToolCallContent, FileEditToolCallContent, TerminalToolCallContent]]],
        Field(description="Content produced by the tool call."),
    ] = None
    # The category of tool being invoked.
    # Helps clients choose appropriate icons and UI treatment.
    kind: Annotated[
        Optional[ToolKind],
        Field(
            description="The category of tool being invoked.\nHelps clients choose appropriate icons and UI treatment."
        ),
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
    # Current execution status of the tool call.
    status: Annotated[Optional[ToolCallStatus], Field(description="Current execution status of the tool call.")] = None
    # Human-readable title describing what the tool is doing.
    title: Annotated[
        str,
        Field(description="Human-readable title describing what the tool is doing."),
    ]
    # Unique identifier for this tool call within the session.
    tool_call_id: Annotated[
        str,
        Field(
            alias="toolCallId",
            description="Unique identifier for this tool call within the session.",
        ),
    ]


class AgentRequest(BaseModel):
    # JSON RPC Request Id
    #
    # An identifier established by the Client that MUST contain a String, Number, or NULL value if included. If it is not included it is assumed to be a notification. The value SHOULD normally not be Null [1] and Numbers SHOULD NOT contain fractional parts [2]
    #
    # The Server MUST reply with the same value in the Response object if included. This member is used to correlate the context between the two objects.
    #
    # [1] The use of Null as a value for the id member in a Request object is discouraged, because this specification uses a value of Null for Responses with an unknown id. Also, because JSON-RPC 1.0 uses an id value of Null for Notifications this could cause confusion in handling.
    #
    # [2] Fractional parts may be problematic, since many decimal fractions cannot be represented exactly as binary fractions.
    id: Annotated[
        Optional[Union[int, str]],
        Field(
            description="JSON RPC Request Id\n\nAn identifier established by the Client that MUST contain a String, Number, or NULL value if included. If it is not included it is assumed to be a notification. The value SHOULD normally not be Null [1] and Numbers SHOULD NOT contain fractional parts [2]\n\nThe Server MUST reply with the same value in the Response object if included. This member is used to correlate the context between the two objects.\n\n[1] The use of Null as a value for the id member in a Request object is discouraged, because this specification uses a value of Null for Responses with an unknown id. Also, because JSON-RPC 1.0 uses an id value of Null for Notifications this could cause confusion in handling.\n\n[2] Fractional parts may be problematic, since many decimal fractions cannot be represented exactly as binary fractions."
        ),
    ] = None
    method: str
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
            Union[CreateFormElicitationRequest, CreateUrlElicitationRequest],
            Any,
        ]
    ] = None


class AgentResponseMessage(BaseModel):
    # JSON RPC Request Id
    #
    # An identifier established by the Client that MUST contain a String, Number, or NULL value if included. If it is not included it is assumed to be a notification. The value SHOULD normally not be Null [1] and Numbers SHOULD NOT contain fractional parts [2]
    #
    # The Server MUST reply with the same value in the Response object if included. This member is used to correlate the context between the two objects.
    #
    # [1] The use of Null as a value for the id member in a Request object is discouraged, because this specification uses a value of Null for Responses with an unknown id. Also, because JSON-RPC 1.0 uses an id value of Null for Notifications this could cause confusion in handling.
    #
    # [2] Fractional parts may be problematic, since many decimal fractions cannot be represented exactly as binary fractions.
    id: Annotated[
        Optional[Union[int, str]],
        Field(
            description="JSON RPC Request Id\n\nAn identifier established by the Client that MUST contain a String, Number, or NULL value if included. If it is not included it is assumed to be a notification. The value SHOULD normally not be Null [1] and Numbers SHOULD NOT contain fractional parts [2]\n\nThe Server MUST reply with the same value in the Response object if included. This member is used to correlate the context between the two objects.\n\n[1] The use of Null as a value for the id member in a Request object is discouraged, because this specification uses a value of Null for Responses with an unknown id. Also, because JSON-RPC 1.0 uses an id value of Null for Notifications this could cause confusion in handling.\n\n[2] Fractional parts may be problematic, since many decimal fractions cannot be represented exactly as binary fractions."
        ),
    ] = None
    # All possible responses that an agent can send to a client.
    #
    # This enum is used internally for routing RPC responses. You typically won't need
    # to use this directly - the responses are handled automatically by the connection.
    #
    # These are responses to the corresponding `ClientRequest` variants.
    result: Annotated[
        Union[
            InitializeResponse,
            AuthenticateResponse,
            ListProvidersResponse,
            SetProvidersResponse,
            DisableProvidersResponse,
            LogoutResponse,
            NewSessionResponse,
            LoadSessionResponse,
            ListSessionsResponse,
            ForkSessionResponse,
            ResumeSessionResponse,
            CloseSessionResponse,
            SetSessionModeResponse,
            SetSessionConfigOptionResponse,
            PromptResponse,
            SetSessionModelResponse,
            StartNesResponse,
            SuggestNesResponse,
            CloseNesResponse,
            Any,
        ],
        Field(
            description="All possible responses that an agent can send to a client.\n\nThis enum is used internally for routing RPC responses. You typically won't need\nto use this directly - the responses are handled automatically by the connection.\n\nThese are responses to the corresponding `ClientRequest` variants."
        ),
    ]


class AgentResponse(RootModel[Union[AgentResponseMessage, AgentErrorMessage]]):
    root: Union[AgentResponseMessage, AgentErrorMessage]


class ToolCallStart(ToolCall):
    session_update: Annotated[Literal["tool_call"], Field(alias="sessionUpdate")]


class SessionNotification(BaseModel):
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
            AvailableCommandsUpdate,
            CurrentModeUpdate,
            ConfigOptionUpdate,
            SessionInfoUpdate,
            UsageUpdate,
        ],
        Field(description="The actual update content.", discriminator="session_update"),
    ]


class AgentNotification(BaseModel):
    method: str
    params: Optional[Union[SessionNotification, CompleteElicitationNotification, Any]] = None
