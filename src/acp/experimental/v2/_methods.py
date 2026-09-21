from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, get_type_hints

from pydantic import TypeAdapter

from .interfaces import Agent, Client


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


@cache
def protocol_specs(protocol: type) -> tuple[tuple[RequestSpec, ...], tuple[NotificationSpec, ...]]:
    """Compile v2 wire validation from the public protocol declarations."""
    requests: dict[str, RequestSpec] = {}
    notifications: dict[str, NotificationSpec] = {}
    for name in dir(protocol):
        handler = getattr(protocol, name)
        metadata = getattr(handler, "__route__", None)
        if metadata is None:
            continue
        if metadata.kind == "notification":
            if metadata.method in notifications:
                raise ValueError(f"Duplicate notification: {metadata.method}")
            notifications[metadata.method] = NotificationSpec(metadata.method, name, TypeAdapter(metadata.request_type))
        else:
            if metadata.method in requests:
                raise ValueError(f"Duplicate request: {metadata.method}")
            requests[metadata.method] = RequestSpec(
                metadata.method,
                name,
                TypeAdapter(metadata.request_type),
                TypeAdapter(get_type_hints(handler, include_extras=True)["return"]),
                empty_response=metadata.default_result == {},
            )
    return tuple(requests.values()), tuple(notifications.values())


AGENT_REQUESTS, AGENT_NOTIFICATIONS = protocol_specs(Agent)
CLIENT_REQUESTS, CLIENT_NOTIFICATIONS = protocol_specs(Client)
AGENT_REQUESTS_BY_METHOD = {spec.method: spec for spec in AGENT_REQUESTS}
CLIENT_REQUESTS_BY_METHOD = {spec.method: spec for spec in CLIENT_REQUESTS}
