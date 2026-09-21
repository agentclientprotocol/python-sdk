"""Protocols are the source of truth for both signatures and wire routing."""

from typing import Any, Literal, Protocol, cast

import pytest
from pydantic import BaseModel, Field, ValidationError

from acp.agent.router import build_agent_router
from acp.client.router import build_client_router
from acp.exceptions import RequestError
from acp.interfaces import Agent, Client
from acp.router import MessageRouter
from acp.schema import SetSessionConfigOptionBooleanRequest
from acp.utils import normalize_result, param_model, param_models


class Params(BaseModel):
    value: str
    field_meta: dict[str, Any] | None = Field(None, alias="_meta")


class Parent(Protocol):
    @param_model(Params, method="example/message", adapt_result=normalize_result)
    async def message(self, value: str, **kwargs: Any) -> Any: ...


class Example(Parent, Protocol):
    @param_model(Params, method="example/message", kind="notification")
    async def notify(self, value: str, **kwargs: Any) -> None: ...

    @param_model(Params, method="example/preview", unstable=True)
    async def preview(self, value: str, **kwargs: Any) -> Any: ...

    @param_model(Params, method="example/optional", optional=True, default_result={})
    async def optional(self, value: str, **kwargs: Any) -> Any: ...

    @param_model(Params)
    async def local(self, value: str, **kwargs: Any) -> Any: ...


@pytest.mark.asyncio
async def test_routes_come_from_protocol_including_inherited_members():
    calls = []

    class Implementation:
        async def message(self, value: str, **kwargs: Any):
            calls.append(("request", value, kwargs))

        async def notify(self, value: str, **kwargs: Any):
            calls.append(("notification", value, kwargs))

        async def preview(self, value: str, **kwargs: Any):
            return value

        @param_model(Params, method="example/private")
        async def private(self, value: str, **kwargs: Any):
            raise AssertionError("Implementation-only routes must not be exposed")

    router = MessageRouter.from_protocol(Example, Implementation())
    assert await router("example/message", {"value": "hello", "_meta": {"trace": 1}}, False) == {}
    assert await router("example/message", {"value": "bye"}, True) is None
    assert calls == [("request", "hello", {"trace": 1}), ("notification", "bye", {})]
    assert await router("example/optional", {}, False) == {}
    with pytest.raises(ValidationError):
        await router("example/message", {}, False)
    for method in ["example/private", "local"]:
        with pytest.raises(RequestError):
            await router(method, {"value": "hidden"}, False)
    with pytest.warns(UserWarning, match="unstable"), pytest.raises(RequestError):
        await router("example/preview", {"value": "preview"}, False)
    enabled = MessageRouter.from_protocol(Example, Implementation(), use_unstable_protocol=True)
    assert await enabled("example/preview", {"value": "preview"}, False) == "preview"


@pytest.mark.asyncio
async def test_union_route_uses_shared_fields_and_preserves_legacy_model():
    class Text(BaseModel):
        value: str
        kind: Literal["text"] = "text"

    class Number(BaseModel):
        value: int
        kind: Literal["number"] = "number"
        precision: int = 0

    class UnionProtocol(Protocol):
        @param_models(Text, Number, method="example/union")
        async def set_value(self, value: str | int, kind: str) -> Any: ...

    class Modern:
        async def set_value(self, value: str | int, kind: str):
            return value, kind

    class Legacy:
        async def setValue(self, params):
            return params

    payload = {"kind": "number", "value": 3, "precision": 2}
    router = MessageRouter.from_protocol(UnionProtocol, Modern())
    assert await router("example/union", payload, False) == (3, "number")
    legacy = MessageRouter.from_protocol(UnionProtocol, Legacy())
    with pytest.warns(DeprecationWarning):
        result = await legacy("example/union", payload, False)
    assert isinstance(result, Number)
    assert result.precision == 2


def test_duplicate_routes_are_rejected():
    class Duplicate(Parent, Protocol):
        @param_model(Params, method="example/message")
        async def other(self, value: str, **kwargs: Any) -> Any: ...

    with pytest.raises(ValueError, match="Duplicate request route"):
        MessageRouter.from_protocol(Duplicate, object())


@pytest.mark.asyncio
async def test_legacy_config_adapter_receives_boolean_request():
    class Legacy:
        async def setConfigOption(self, params):
            assert isinstance(params, SetSessionConfigOptionBooleanRequest)
            assert params.value is False
            return None

    router = build_agent_router(cast(Agent, Legacy()))
    with pytest.warns(DeprecationWarning):
        assert (
            await router(
                "session/set_config_option",
                {"sessionId": "s", "configId": "flag", "type": "boolean", "value": False},
                False,
            )
            == {}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_elicitation_custom_mode_keeps_existing_dispatch_behavior(legacy):
    class Handler:
        async def createElicitation(self, params):
            return {"mode": params.mode}

    class Modern:
        async def create_elicitation(self, **kwargs):
            raise AssertionError("Unsupported modes must fail before invoking the handler")

    router = build_client_router(cast(Client, Handler() if legacy else Modern()), use_unstable_protocol=True)
    payload = {"mode": "x-voice", "message": "hi", "sessionId": "s"}
    if legacy:
        with pytest.warns(DeprecationWarning):
            assert await router("elicitation/create", payload, False) == {"mode": "x-voice"}
    else:
        with pytest.raises(RequestError) as error:
            await router("elicitation/create", payload, False)
        assert isinstance(error.value, RequestError)
        assert error.value.code == -32602


@pytest.mark.asyncio
async def test_missing_handlers_and_extensions():
    router = MessageRouter.from_protocol(Example, object())
    with pytest.raises(RequestError):
        await router("example/message", {"value": "hi"}, False)
    with pytest.raises(RequestError):
        await router("_example/extension", {}, False)
    assert await router("_example/extension", {}, True) is None

    class Extensions:
        async def ext_method(self, name, payload):
            return [name, payload]

        async def ext_notification(self, name, payload):
            assert name == "example/extension"
            assert payload == {"value": 1}

    router = MessageRouter.from_protocol(Example, Extensions())
    assert await router("_example/extension", {"value": 1}, False) == ["example/extension", {"value": 1}]
    assert await router("_example/extension", {"value": 1}, True) is None
