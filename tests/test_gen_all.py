from pathlib import Path

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
