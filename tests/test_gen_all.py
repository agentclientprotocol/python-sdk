from scripts.gen_all import resolve_ref, schema_source_paths


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
