from __future__ import annotations

import sys
import typing
import dataclasses
import builtins
from typing import *

if sys.version_info < (3, 9):
    from typing import (
        Dict as dict,
        List as list,
        Tuple as tuple,
        Set as set,
        Type as type,
    )
    import typing_extensions
    from typing_extensions import Annotated, Literal, Protocol, runtime_checkable, TypedDict, TypeAlias, Final
else:
    # Ensure they are available for import
    list = builtins.list
    dict = builtins.dict
    tuple = builtins.tuple
    set = builtins.set
    type = builtins.type
    from typing import Annotated, Literal, Protocol, runtime_checkable, TypedDict, TypeAlias, Final

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec, TypeGuard
    # Union can't be used with | syntax anyway, but we export it for convenience
    Union = typing.Union
    Optional = typing.Optional
else:
    from typing import ParamSpec, TypeGuard
    Union = typing.Union
    Optional = typing.Optional

def dataclass_with_slots(*args, **kwargs):
    if sys.version_info < (3, 10):
        kwargs.pop("slots", None)
    return dataclasses.dataclass(*args, **kwargs)
