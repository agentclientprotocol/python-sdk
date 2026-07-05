from scripts.gen_all import resolve_ref, schema_source_paths
from scripts.gen_schema import _preprocess_schema_for_codegen, _restore_required_nullable_fields


def test_resolve_ref_accepts_schema_release_tags() -> None:
    assert resolve_ref("schema-v1.16.0") == "refs/tags/schema-v1.16.0"


def test_resolve_ref_keeps_legacy_version_tags() -> None:
    assert resolve_ref("0.13.6") == "refs/tags/v0.13.6"
    assert resolve_ref("v0.13.6") == "refs/tags/v0.13.6"


def test_schema_release_tags_prefer_v1_schema_layout() -> None:
    assert schema_source_paths("refs/tags/schema-v1.16.0")[0] == (
        "schema/v1/schema.unstable.json",
        "schema/v1/meta.unstable.json",
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


def test_codegen_preprocess_distributes_common_object_properties() -> None:
    schema = {
        "$defs": {
            "ScopeA": {
                "type": "object",
                "properties": {"scopeA": {"type": "string"}},
                "required": ["scopeA"],
            },
            "ScopeB": {
                "type": "object",
                "properties": {"scopeB": {"type": "string"}},
                "required": ["scopeB"],
            },
            "Mode": {
                "type": "object",
                "properties": {"payload": {"type": "string"}},
                "required": ["payload"],
                "anyOf": [
                    {"allOf": [{"$ref": "#/$defs/ScopeA"}]},
                    {"allOf": [{"$ref": "#/$defs/ScopeB"}]},
                ],
            },
            "Request": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"kind": {"type": "string", "const": "mode"}},
                        "required": ["kind"],
                        "allOf": [{"$ref": "#/$defs/Mode"}],
                    }
                ],
            },
        },
        "$ref": "#/$defs/Request",
    }

    request = _preprocess_schema_for_codegen(schema)["$defs"]["Request"]

    assert len(request["oneOf"]) == 2
    assert request["oneOf"][0]["required"] == ["message", "kind", "payload"]
    assert request["oneOf"][0]["properties"].keys() >= {"message", "kind", "payload"}
    assert request["oneOf"][0]["allOf"] == [{"$ref": "#/$defs/ScopeA"}]
    assert request["oneOf"][1]["allOf"] == [{"$ref": "#/$defs/ScopeB"}]


def test_codegen_postprocess_preserves_required_nullable_fields() -> None:
    schema = {
        "$defs": {
            "Example": {
                "type": "object",
                "properties": {
                    "requiredId": {"anyOf": [{"type": "null"}, {"type": "string"}]},
                    "optionalId": {"anyOf": [{"type": "null"}, {"type": "string"}]},
                },
                "required": ["requiredId"],
            }
        }
    }
    content = """\
class Example(BaseModel):
    required_id: Annotated[
        Optional[str],
        Field(alias="requiredId"),
    ] = None
    optional_id: Annotated[
        Optional[str],
        Field(alias="optionalId"),
    ] = None
"""

    processed = _restore_required_nullable_fields(content, schema)

    assert 'Field(alias="requiredId"),\n    ] = None' not in processed
    assert 'Field(alias="requiredId"),\n    ]' in processed
    assert 'Field(alias="optionalId"),\n    ] = None' in processed
