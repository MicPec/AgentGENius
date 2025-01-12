from types import NoneType
from typing import Annotated, Any, GenericAlias, Literal, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic._internal._model_construction import ModelMetaclass
from pydantic._internal._schema_generation_shared import GetJsonSchemaHandler
from pydantic_ai.agent import EndStrategy, RunContext
from pydantic_ai.models import KnownModelName
from pydantic_ai.result import ResultData
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDeps
from pydantic_core import CoreSchema, core_schema

from agentgenius.utils import search_frame

TYPE_MAPPING = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
    "bytes": bytes,
    "bytearray": bytearray,
    "None": None,
    "NoneType": type(None),
    # "TaskDef": "TaskDef",  # Will be resolved dynamically
}


class TypeField:
    @classmethod
    def validate(cls, value: Union[str, type, GenericAlias, NoneType]) -> Type:
        """Validate and convert type field values"""
        if isinstance(value, str):
            if value in TYPE_MAPPING:
                return TYPE_MAPPING[value]
            # Try to evaluate as a type expression
            try:
                return eval(value, search_frame(value))
            except Exception as e:
                raise ValueError(f"Invalid type: {value}") from e
        if isinstance(value, (type, GenericAlias, ModelMetaclass)):
            return value
        raise ValueError(f"Expected type or type name, got {type(value)}")

    @classmethod
    def serialize(cls, value: Type) -> str:
        """Serialize type to string representation"""
        if value is None:
            return "None"
        if isinstance(value, GenericAlias):
            origin = get_origin(value)
            args = get_args(value)
            return f"{origin.__name__}[{', '.join(cls.serialize(arg) for arg in args)}]"
        return value.__name__ if hasattr(value, "__name__") else str(value)

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: CoreSchema,
        _handler: GetJsonSchemaHandler,
    ) -> dict[str, Any]:
        """Generate JSON schema for type fields"""
        return {
            "type": "string",
            "description": "Python type name (e.g., 'str', 'int', 'list[str]')",
            "examples": ["str", "int", "list[str]", "TaskDef"],
        }

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Any,
    ) -> CoreSchema:
        """Generate core schema for validation and serialization"""
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(type),
                    core_schema.is_instance_schema(GenericAlias),
                    core_schema.str_schema(),
                    core_schema.int_schema(),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(cls.serialize),
        )


class AgentParams(BaseModel):
    result_type: Optional[Annotated[Type, TypeField]] = Field(default=str)
    deps_type: Optional[Annotated[Type, TypeField]] = Field(default=NoneType)
    model_settings: Optional[dict] = Field(default=None)
    retries: int = Optional[Field(default=1)]
    result_tool_name: Optional[str] = Field(default="final_result")
    result_tool_description: Optional[str] = Field(default=None)
    result_retries: Optional[int] = Field(default=None)
    defer_model_check: Optional[bool] = Field(default=False)
    end_strategy: Optional[EndStrategy] = Field(default="early")

    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        json_encoders={type: TypeField.serialize},
    )

    @field_validator("result_type", "deps_type", mode="plain")
    @classmethod
    def validate_type(cls, value):
        result = TypeField.validate(value)
        return result


class AgentDef(BaseModel):
    model: KnownModelName
    name: str
    system_prompt: str
    params: Optional[AgentParams] = Field(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={type: TypeField.serialize},
        # defer_build=True,
    )
