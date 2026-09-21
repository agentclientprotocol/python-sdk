from pathlib import Path

import pytest

from acp.schema import ReadTextFileRequest
from scripts.gen_all import resolve_ref, schema_source_paths
from scripts.gen_meta import generate_meta
from scripts.gen_schema import generate_schema


def test_generated_field_descriptions_are_introspectable() -> None:
    path_description = "Absolute path to the file to read."
    assert ReadTextFileRequest.model_fields["path"].description == path_description
    assert ReadTextFileRequest.model_json_schema()["properties"]["path"]["description"] == path_description


def test_resolve_ref_accepts_schema_release_tags() -> None:
    assert resolve_ref("schema-v1.16.0") == "refs/tags/schema-v1.16.0"
    assert resolve_ref("schema-v2.0.0-alpha.3") == "refs/tags/schema-v2.0.0-alpha.3"


def test_resolve_ref_keeps_legacy_version_tags() -> None:
    assert resolve_ref("0.13.6") == "refs/tags/v0.13.6"
    assert resolve_ref("v0.13.6") == "refs/tags/v0.13.6"


def test_schema_release_tags_prefer_v1_schema_layout() -> None:
    assert schema_source_paths("refs/tags/schema-v1.16.0")[0] == (
        "schema/v1/schema.unstable.json",
        "schema/v1/meta.unstable.json",
    )


def test_v2_generation_uses_v2_schema_layout() -> None:
    assert schema_source_paths("refs/tags/schema-v2.0.0-alpha.3", 2) == (
        ("schema/v2/schema.unstable.json", "schema/v2/meta.unstable.json"),
    )


def test_legacy_tags_keep_legacy_schema_layout_first() -> None:
    assert schema_source_paths("refs/tags/v0.13.6")[0] == (
        "schema/schema.unstable.json",
        "schema/meta.unstable.json",
    )


def test_parse_args_formats_output_by_default(monkeypatch) -> None:
    from scripts import gen_all

    monkeypatch.setattr("sys.argv", ["gen_all.py"])
    assert gen_all.parse_args().format_output is True


def test_parse_args_can_skip_format(monkeypatch) -> None:
    from scripts import gen_all

    monkeypatch.setattr("sys.argv", ["gen_all.py", "--no-format"])
    assert gen_all.parse_args().format_output is False


def test_codegen_check_is_clean_and_read_only() -> None:
    outputs = (
        Path("src/acp/meta.py"),
        Path("src/acp/schema.py"),
        Path("src/acp/experimental/v2/meta.py"),
        Path("src/acp/experimental/v2/schema.py"),
    )
    before = {output: output.read_bytes() for output in outputs}

    assert generate_schema(check=True)
    assert generate_meta(check=True)
    assert generate_schema(check=True, protocol_version=2)
    assert generate_meta(check=True, protocol_version=2)
    assert {output: output.read_bytes() for output in outputs} == before


def test_signature_generation_preserves_inline_literal_values() -> None:
    import ast
    from typing import Literal, get_type_hints

    from scripts.gen_signature import NodeTransformer

    tree = ast.parse(
        "from typing import Any\n"
        "from schema import SetProviderRequest, SuggestNesRequest\n"
        "class Methods:\n"
        '    @param_model(SetProviderRequest, method="providers/set", unstable=True)\n'
        "    async def set_provider(self, **kwargs: Any): ...\n"
        "    @param_model(SuggestNesRequest)\n"
        "    async def suggest_nes(self, **kwargs: Any): ...\n"
    )
    NodeTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    # Evaluate the annotation itself to catch unquoted literal values and ensure
    # generation adds the required typing import.
    typing_import = tree.body[0]
    assert isinstance(typing_import, ast.ImportFrom)
    assert "Literal" in {alias.name for alias in typing_import.names}
    methods = tree.body[-1]
    assert isinstance(methods, ast.ClassDef)
    provider = methods.body[0]
    assert isinstance(provider, ast.AsyncFunctionDef)
    decorator = provider.decorator_list[0]
    assert isinstance(decorator, ast.Call)
    assert {keyword.arg: ast.literal_eval(keyword.value) for keyword in decorator.keywords} == {
        "method": "providers/set",
        "unstable": True,
    }
    suggest = methods.body[-1]
    assert isinstance(suggest, ast.AsyncFunctionDef)
    trigger = next(arg for arg in suggest.args.args if arg.arg == "trigger_kind")
    annotation = ast.unparse(trigger.annotation)

    class Annotated:
        __annotations__ = {"trigger": annotation}

    assert (
        get_type_hints(Annotated, globalns={"Literal": Literal})["trigger"]
        == Literal["automatic", "diagnostic", "manual"]
    )


@pytest.mark.parametrize(
    "expression",
    [
        "SetSessionConfigOptionBooleanRequest | SetSessionConfigOptionSelectRequest",
        "Union[SetSessionConfigOptionBooleanRequest, SetSessionConfigOptionSelectRequest]",
        "Annotated[SetSessionConfigOptionBooleanRequest | SetSessionConfigOptionSelectRequest, Field(discriminator='type')]",
        "RequestAlias",
        "CreateElicitationRequest",
    ],
)
def test_signature_generation_preserves_union_signatures(expression) -> None:
    import ast

    from scripts.gen_signature import NodeTransformer

    tree = ast.parse(
        "from typing import Annotated, Any, Union\n"
        "from schema import SetSessionConfigOptionBooleanRequest, SetSessionConfigOptionSelectRequest, CreateElicitationRequest\n"
        "RequestAlias = SetSessionConfigOptionBooleanRequest | SetSessionConfigOptionSelectRequest\n"
        "class Methods:\n"
        f"    @param_model({expression}, method='example/request')\n"
        "    async def request(self, config_id: str, session_id: str, value: str | bool, **kwargs: Any): ...\n"
    )
    before = ast.dump(tree)
    NodeTransformer().visit(tree)
    assert ast.dump(tree) == before


@pytest.mark.parametrize(
    "expression",
    [
        "DeleteSessionRequest",
        "Annotated[DeleteSessionRequest, Field(title='Delete')]",
        "RequestAlias",
        "schema.DeleteSessionRequest",
    ],
)
def test_signature_generation_expands_single_model_types(expression) -> None:
    import ast

    from scripts.gen_signature import NodeTransformer

    tree = ast.parse(
        "from typing import Annotated, Any, TypeAlias\n"
        "from schema import DeleteSessionRequest\n"
        "RequestAlias: TypeAlias = Annotated[DeleteSessionRequest, Field(title='Delete')]\n"
        "class Methods:\n"
        f"    @param_model({expression}, method='session/delete')\n"
        "    async def delete_session(self, **kwargs: Any): ...\n"
    )
    NodeTransformer().visit(tree)
    methods = tree.body[-1]
    assert isinstance(methods, ast.ClassDef)
    method = methods.body[0]
    assert isinstance(method, ast.AsyncFunctionDef)
    assert [arg.arg for arg in method.args.args] == ["self", "session_id"]
    assert ast.unparse(method.decorator_list[0]) == f"param_model({expression}, method='session/delete')"
