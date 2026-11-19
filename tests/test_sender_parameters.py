"""Every sender puts each of its parameters on the wire.

`scripts/gen_signature.py` rewrites sender signatures from the schema, but the bodies are hand-written. A field the
schema adds can land in a signature and never reach the request model, and the call then drops it silently.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "acp"


@pytest.mark.parametrize("module", ["client/connection.py", "agent/connection.py"])
def test_every_sender_uses_all_its_parameters(module: str) -> None:
    unused: list[str] = []
    for node in ast.walk(ast.parse((SRC / module).read_text(encoding="utf-8"))):
        if isinstance(node, ast.AsyncFunctionDef) and not node.name.startswith("__"):
            params = [a.arg for a in node.args.args + node.args.kwonlyargs if a.arg != "self"]
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            unused += [f"{node.name}({p})" for p in params if p not in names]
    assert unused == []
