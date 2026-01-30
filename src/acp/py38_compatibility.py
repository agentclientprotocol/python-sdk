from __future__ import annotations

import sys
import typing
import dataclasses
import builtins
from typing import *

if sys.version_info < (3, 9):
    import builtins
    import typing
    import typing_extensions
    from typing_extensions import Annotated, Literal, Protocol, runtime_checkable, TypedDict, TypeAlias, Final

    def _make_generic_constructor(origin, typing_type):
        class Meta(type):
            def __getitem__(cls, item):
                return typing_type[item]
            def __instancecheck__(cls, instance):
                return isinstance(instance, origin)
            def __subclasscheck__(cls, subclass):
                return issubclass(subclass, origin)

        class Constructor(metaclass=Meta):
            def __new__(cls, *args, **kwargs):
                return origin(*args, **kwargs)

        Constructor.__name__ = origin.__name__
        Constructor.__module__ = origin.__module__
        return Constructor

    list = _make_generic_constructor(builtins.list, typing.List)
    dict = _make_generic_constructor(builtins.dict, typing.Dict)
    tuple = _make_generic_constructor(builtins.tuple, typing.Tuple)
    set = _make_generic_constructor(builtins.set, typing.Set)
    type = _make_generic_constructor(builtins.type, typing.Type)

    # asyncio generics compatibility
    import asyncio
    class _AsyncioGenericAlias:
        def __init__(self, origin):
            self._origin = origin
        def __getitem__(self, _):
            return self._origin
        def __instancecheck__(self, instance):
            return isinstance(instance, self._origin)
        def __subclasscheck__(self, subclass):
            return issubclass(subclass, self._origin)

    Task = _AsyncioGenericAlias(asyncio.Task)
    Future = _AsyncioGenericAlias(asyncio.Future)
    Queue = _AsyncioGenericAlias(asyncio.Queue)
    StreamReader = _AsyncioGenericAlias(asyncio.StreamReader)
    StreamWriter = _AsyncioGenericAlias(asyncio.StreamWriter)

else:
    # Ensure they are available for import
    list = builtins.list
    dict = builtins.dict
    tuple = builtins.tuple
    set = builtins.set
    type = builtins.type
    from typing import Annotated, Literal, Protocol, runtime_checkable, TypedDict, TypeAlias, Final
    import asyncio
    Task = asyncio.Task
    Future = asyncio.Future
    Queue = asyncio.Queue
    StreamReader = asyncio.StreamReader
    StreamWriter = asyncio.StreamWriter

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
