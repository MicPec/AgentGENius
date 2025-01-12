import inspect
from types import GenericAlias
from typing import Any

from pydantic import TypeAdapter


class TypeAdapterMixin:
    @classmethod
    def model_dump_json(cls, instance: object, **kwargs):
        adapter = TypeAdapter(cls)
        return adapter.dump_json(instance, **kwargs)

    @classmethod
    def model_dump(cls, instance: object, **kwargs):
        adapter = TypeAdapter(cls)
        return adapter.dump_python(instance, **kwargs)


def custom_type_encoder(obj: Any) -> Any:
    if isinstance(obj, type):
        return obj.__name__  # Serialize type objects as their names
    if isinstance(obj, GenericAlias):
        return str(obj)
    return obj.dict()  # Fallback to default Pydantic encoder


def search_frame(value: str):
    frame = inspect.currentframe()
    while frame:
        result = frame.f_locals.get(value)
        if result is not None:
            return frame.f_locals | frame.f_globals
        result = frame.f_globals.get(value)
        if result is not None:
            return frame.f_globals
        frame = frame.f_back
    raise ValueError(f"'{value}' not found")
