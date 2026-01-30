import os
import re

def add_compat_imports(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and file != "py38_compatibility.py":
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()

                # Check if it's in a package that can reach py38_compatibility
                # For tests, it needs from acp.py38_compatibility import *

                if "from __future__ import annotations" in content and "py38_compatibility" not in content:
                    if "tests" in root:
                        import_line = "from acp.py38_compatibility import *"
                    elif "examples" in root:
                         import_line = "from acp.py38_compatibility import *"
                    else:
                         # For src/acp/... it depends on depth
                         depth = len(root.split(os.sep)) - len("src/acp".split(os.sep))
                         dots = "." * (depth + 1)
                         import_line = f"from {dots}py38_compatibility import *"

                    content = content.replace("from __future__ import annotations", f"from __future__ import annotations\n{import_line}")

                with open(filepath, 'w') as f:
                    f.write(content)

add_compat_imports("tests")
