from __future__ import annotations

import attrs
import cattrs
import builtins
from typing import Any, Type, TypeVar, Dict, Optional, List, Union, Generic
from inspect import isclass
import sys
import re
from enum import Enum
import functools

from .py38_compatibility import list, dict, Annotated, get_type_hints

T = TypeVar("T", bound="BaseModel")
RT = TypeVar("RT")

@functools.lru_cache(maxsize=None)
def _get_cached_type_hints(cls):
    return get_type_hints(cls, include_extras=True)

class FieldInfo:
    def __init__(self, default=attrs.NOTHING, alias=None, description=None, ge=None, le=None, **kwargs):
        self.default = default
        self.alias = alias
        self.description = description
        self.ge = ge
        self.le = le
        self.kwargs = kwargs

def Field(default=attrs.NOTHING, **kwargs):
    return FieldInfo(default=default, **kwargs)

class ConfigDict(dict):
    pass

class ValidationError(Exception):
    def __init__(self, errors=None):
        self._errors = errors or []
    def errors(self):
        return self._errors
    def __str__(self):
        return f"ValidationError: {self._errors}"

# Use a converter that enforces types
converter = cattrs.Converter()

def structure_int(obj, cls):
    if not isinstance(obj, int) or isinstance(obj, bool):
        raise ValueError(f"Expected int, got {type(obj)}")
    return obj

def structure_str(obj, cls):
    if not isinstance(obj, str):
        raise ValueError(f"Expected str, got {type(obj)}")
    return obj

def structure_bool(obj, cls):
    if not isinstance(obj, bool):
        raise ValueError(f"Expected bool, got {type(obj)}")
    return obj

def structure_float(obj, cls):
    if not isinstance(obj, (int, float)) or isinstance(obj, bool):
        raise ValueError(f"Expected float, got {type(obj)}")
    return float(obj)

converter.register_structure_hook(int, structure_int)
converter.register_structure_hook(str, structure_str)
converter.register_structure_hook(bool, structure_bool)
converter.register_structure_hook(float, structure_float)

def structure_literal(obj, cls):
    if obj not in cls.__args__:
        raise ValueError(f"Expected one of {cls.__args__}, got {obj!r}")
    return obj

from cattrs.converters import is_literal as cattrs_is_literal
converter.register_structure_hook_func(cattrs_is_literal, structure_literal)

def _structure_acp_model(obj, cls):
    if not isinstance(obj, dict):
        if isinstance(obj, cls):
            return obj
        raise ValueError(f"Expected dict for {cls}, got {type(obj)}")

    try:
        hints = _get_cached_type_hints(cls)
    except Exception:
        hints = {}

    mapped_obj = {}
    errors = []
    for field in attrs.fields(cls):
        alias = field.metadata.get("alias")
        val = None
        found = False
        if alias and alias in obj:
            val = obj[alias]
            found = True
        elif field.name in obj:
            val = obj[field.name]
            found = True

        if found:
            field_type = hints.get(field.name)
            if field_type:
                if hasattr(field_type, "__origin__") and field_type.__origin__ is Annotated:
                     field_type = field_type.__args__[0]

                try:
                    res = converter.structure(val, field_type)
                    mapped_obj[field.name] = res
                except (ValueError, TypeError, ValidationError) as e:
                    errors.append({"loc": [alias or field.name], "msg": str(e), "type": "type_error"})
            else:
                mapped_obj[field.name] = val
        elif field.default is attrs.NOTHING:
            field_type = hints.get(field.name)
            is_opt = False
            if field_type:
                if hasattr(field_type, "__origin__") and field_type.__origin__ is Annotated:
                     field_type = field_type.__args__[0]
                origin = getattr(field_type, "__origin__", None)
                if origin is Union:
                    if type(None) in field_type.__args__:
                        is_opt = True
                elif sys.version_info >= (3, 10):
                    import types
                    if isinstance(origin, types.UnionType):
                         if type(None) in field_type.__args__:
                            is_opt = True

            if not is_opt:
                errors.append({"loc": [alias or field.name], "msg": "field required", "type": "value_error.missing"})

    if errors:
        raise ValidationError(errors)

    return cls(**mapped_obj)

def _structure_root_model(obj, cls):
    try:
        hints = _get_cached_type_hints(cls)
        root_type = hints.get("root")
    except Exception:
        root_type = None

    if root_type:
        if isinstance(obj, dict) and "root" in obj:
            return cls(root=converter.structure(obj["root"], root_type))
        return cls(root=converter.structure(obj, root_type))
    return cls(root=obj)

class BaseModelMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        if name in ("BaseModel", "RootModel", "_BaseModel"):
            return super().__new__(mcs, name, bases, namespace)

        annotations = namespace.get("__annotations__", {})
        for field_name in builtins.list(annotations.keys()):
            hint = annotations[field_name]
            alias = None
            if isinstance(hint, str):
                m = re.search(r'alias=["\']([^"\']+)["\']', hint)
                if m:
                    alias = m.group(1)
            elif hasattr(hint, "__metadata__"):
                 for meta in hint.__metadata__:
                     if isinstance(meta, FieldInfo) and meta.alias:
                         alias = meta.alias
                         break

            default = attrs.NOTHING
            if field_name in namespace:
                val = namespace[field_name]
                if isinstance(val, FieldInfo):
                    if alias is None: alias = val.alias
                    default = val.default
                else:
                    default = val

            namespace[field_name] = attrs.field(default=default, metadata={"alias": alias}, kw_only=True)

        cls = super().__new__(mcs, name, bases, namespace)
        cls = attrs.define(cls, slots=False, init=True, kw_only=True)
        cls.model_fields = {field.name: field for field in attrs.fields(cls)}

        orig_init = cls.__init__
        def __init__(self, **kwargs):
            self.__pydantic_fields_set__ = builtins.set(kwargs.keys())
            orig_init(self, **kwargs)
        cls.__init__ = __init__

        return cls

class BaseModel(metaclass=BaseModelMeta):
    model_fields = {}

    @classmethod
    def model_validate(cls: Type[T], obj: Any) -> T:
        if isinstance(obj, cls):
            return obj
        try:
            return converter.structure(obj, cls)
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError([{"msg": str(e)}]) from e

    def model_dump(self, *, mode: str = "python", by_alias: bool = False, exclude_none: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False) -> Dict[str, Any]:
        res = {}
        fields_set = getattr(self, "__pydantic_fields_set__", builtins.set())
        for field in attrs.fields(self.__class__):
            if exclude_unset and field.name not in fields_set:
                 continue

            val = getattr(self, field.name)
            if exclude_none and val is None:
                continue
            if exclude_defaults and val == field.default:
                continue

            alias = field.metadata.get("alias")
            name = alias if by_alias and alias else field.name
            res[name] = self._dump_value(val, mode, by_alias, exclude_none, exclude_defaults, exclude_unset)
        return res

    def _dump_value(self, val, mode, by_alias, exclude_none, exclude_defaults, exclude_unset):
        if hasattr(val, "model_dump"):
            return val.model_dump(mode=mode, by_alias=by_alias, exclude_none=exclude_none, exclude_defaults=exclude_defaults, exclude_unset=exclude_unset)
        if isinstance(val, builtins.list):
            return [self._dump_value(item, mode, by_alias, exclude_none, exclude_defaults, exclude_unset) for item in val]
        if isinstance(val, builtins.dict):
            return {k: self._dump_value(v, mode, by_alias, exclude_none, exclude_defaults, exclude_unset) for k, v in val.items()}
        if mode == "json" and isinstance(val, Enum):
            return val.value
        return val

    def model_copy(self, *, deep: bool = False):
        import copy
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)

class RootModel(BaseModel, Generic[RT]):
    root: RT

    def __init__(self, root: RT = None, **kwargs):
        if root is not None:
            self.root = root
            self.__pydantic_fields_set__ = {"root"}
        elif "root" in kwargs:
            self.root = kwargs["root"]
            self.__pydantic_fields_set__ = {"root"}
        else:
             self.__pydantic_fields_set__ = builtins.set()

    @classmethod
    def model_validate(cls: Type[T], obj: Any) -> T:
        try:
            return converter.structure(obj, cls)
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError([{"msg": str(e)}]) from e

    def model_dump(self, **kwargs) -> Any:
        val = self.root
        if hasattr(val, "model_dump"):
            return val.model_dump(**kwargs)
        res = converter.unstructure(val)
        if kwargs.get("exclude_none") and res is None:
             return None
        return res

def is_base_model(cls):
    if not isclass(cls):
        return False
    try:
        return issubclass(cls, BaseModel) and not issubclass(cls, RootModel)
    except:
        return False

def is_root_model(cls):
    if not isclass(cls):
        return False
    try:
        return issubclass(cls, RootModel) and cls is not RootModel
    except:
        return False

converter.register_structure_hook_func(is_base_model, _structure_acp_model)
converter.register_structure_hook_func(is_root_model, _structure_root_model)

def is_union(cls):
    origin = getattr(cls, "__origin__", None)
    if origin is Union:
        return True
    if sys.version_info >= (3, 10):
        import types
        if isinstance(origin, types.UnionType) or origin is types.UnionType:
            return True
    return False

import cattrs.errors
def structure_union(obj, cls):
    if obj is None:
        for arg in cls.__args__:
            if arg is type(None):
                return None

    args = sorted(cls.__args__, key=lambda x: 0 if is_root_model(x) else (1 if is_base_model(x) else 2))

    for arg in args:
        if arg is type(None):
            continue
        try:
            return converter.structure(obj, arg)
        except (ValueError, TypeError, ValidationError, cattrs.errors.BaseValidationError):
            continue
    raise ValidationError([{"msg": f"Value {obj} does not match any member of {cls}"}])

converter.register_structure_hook_func(is_union, structure_union)
