from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROTOCOL_VERSION = 1
SEMANTIC_MODULES = {
    1: "scripts.gen_schema_v1",
    2: "scripts.gen_schema_v2",
}
SUPPORTED_PROTOCOL_VERSIONS = tuple(SEMANTIC_MODULES)


@dataclass(frozen=True, slots=True)
class SchemaSemantics:
    schema_json: Path
    version_file: Path
    schema_out: Path
    base_class: str
    model_name_map: dict[str, str]
    compatibility_aliases: str = ""


class _SemanticsModule(Protocol):
    SEMANTICS: SchemaSemantics


def get_schema_semantics(protocol_version: int) -> SchemaSemantics:
    try:
        module_name = SEMANTIC_MODULES[protocol_version]
    except KeyError:
        raise ValueError(f"Unsupported protocol version: {protocol_version}") from None
    module = cast(_SemanticsModule, import_module(module_name))
    return module.SEMANTICS


def inline_model_ref(definition: str, *steps: tuple[str, int | None]) -> str:
    ref = f"#/$defs/{definition}"
    for keyword, index in steps:
        ref += f"#-datamodel-code-generator-#-{keyword}-#-special-#"
        if index is not None:
            ref += f"/{index}"
    return ref


def variant_model_map(
    definition: str,
    keyword: str,
    branch: str,
    names: tuple[str, ...],
) -> dict[str, str]:
    return {inline_model_ref(definition, (keyword, index), (branch, None)): name for index, name in enumerate(names)}
