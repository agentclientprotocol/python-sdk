from typing import Annotated, Any, Literal, Protocol, Union

import pytest
from pydantic import BaseModel, BeforeValidator, Field, ValidationError

from acp.router import MessageRouter
from acp.utils import compatible_class, param_model


class TextRequest(BaseModel):
    kind: Literal["text"] = "text"
    value: str
    field_meta: dict[str, Any] | None = Field(None, alias="_meta")


class NumberRequest(BaseModel):
    kind: Literal["number"] = "number"
    value: int
    precision: int = 0
    field_meta: dict[str, Any] | None = Field(None, alias="_meta")


TaggedRequest = Annotated[TextRequest | NumberRequest, Field(discriminator="kind")]


@pytest.mark.parametrize(
    "request_type",
    [
        TextRequest,
        TextRequest | NumberRequest,
        Union[TextRequest, NumberRequest],  # noqa: UP007 - cover the typing.Union spelling
        TaggedRequest,
        Annotated[TextRequest, Field(title="Text")],
        Annotated[TextRequest, Field(title="Text")] | NumberRequest,
    ],
)
def test_decorator_preserves_original_type_and_function(request_type):
    async def handler(**kwargs):
        return kwargs

    decorated = param_model(request_type, method="example/request")(handler)
    assert decorated is handler
    assert decorated.__param_model__ is request_type
    assert decorated.__route__.request_type is request_type
    assert not hasattr(decorated, "__param_models__")


@pytest.mark.parametrize("request_type", [str, Any, list[TextRequest], TextRequest | str, TextRequest | None])
def test_decorator_rejects_non_model_branches(request_type):
    with pytest.raises(TypeError, match="expects a BaseModel"):
        param_model(request_type)


def test_decorator_accepts_only_one_type_expression():
    with pytest.raises(TypeError):
        param_model(TextRequest, NumberRequest)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_discriminated_union_keeps_validation_metadata_and_shared_fields():
    class Interface(Protocol):
        @param_model(TaggedRequest, method="example/request")
        async def request(self, value: str | int, kind: str, **kwargs: Any) -> Any: ...

    class Handler:
        async def request(self, value: str | int, kind: str, **kwargs: Any):
            return value, kind, kwargs

    router = MessageRouter.from_protocol(Interface, Handler())
    assert await router(
        "example/request",
        {
            "kind": "number",
            "value": 3,
            "precision": 2,
            "_meta": {"trace": "test"},
        },
        False,
    ) == (3, "number", {"trace": "test"})
    # An ordinary union would infer a branch from defaults. A tagged union must
    # require its discriminator, proving the Annotated metadata was retained.
    with pytest.raises(ValidationError) as error:
        await router("example/request", {"value": "hello"}, False)
    assert isinstance(error.value, ValidationError)
    assert error.value.errors()[0]["type"] == "union_tag_not_found"


@pytest.mark.asyncio
async def test_annotated_single_model_keeps_custom_validator():
    def uppercase(payload):
        return {**payload, "value": payload["value"].upper()}

    class Interface(Protocol):
        @param_model(Annotated[TextRequest, BeforeValidator(uppercase)], method="example/request")
        async def request(self, value: str, kind: str, **kwargs: Any) -> str: ...

    class Handler:
        async def request(self, value: str, kind: str, **kwargs: Any):
            return value

    router = MessageRouter.from_protocol(Interface, Handler())
    assert await router("example/request", {"value": "hello"}, False) == "HELLO"


@pytest.mark.parametrize("request_type", [TextRequest | NumberRequest, TaggedRequest])
def test_union_legacy_calls_use_shared_fields_and_meta(request_type):
    @compatible_class
    class Calls:
        @param_model(request_type)
        def set_value(self, value, kind, **kwargs):
            return value, kind, kwargs

        @param_model(request_type)
        def configure(self, value, kind, **kwargs):
            return value, kind, kwargs

    calls = Calls()
    request = NumberRequest(value=4, precision=2, _meta={"trace": "test"})
    expected = (4, "number", {"trace": "test"})
    with pytest.warns(DeprecationWarning):
        assert getattr(calls, "setValue")(request) == expected  # noqa: B009 - dynamically added alias
    with pytest.warns(DeprecationWarning):
        assert calls.configure(request) == expected
    with pytest.warns(DeprecationWarning):
        assert calls.configure(params=request) == expected
    assert calls.configure(value=4, kind="number", trace="test") == expected
