from types import NoneType
from typing import (
    Annotated,
    Any,
    Dict,
    GenericAlias,
    Optional,
    Type,
    Union,
    _GenericAlias,
    _UnionGenericAlias,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

# from pydantic._internal._model_construction import ModelMetaclass
from pydantic._internal._schema_generation_shared import GetJsonSchemaHandler
from pydantic_ai.agent import EndStrategy
from pydantic_ai.models import Model

# from pydantic_ai.result import ResultData
from pydantic_core import CoreSchema, core_schema

from agentgenius.config import config
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
}


class TypeField:
    @classmethod
    def validate(cls, value: Union[str, type, GenericAlias, _GenericAlias, _UnionGenericAlias, NoneType]) -> Type:
        """Validate and convert type field values"""
        if isinstance(value, str):
            if value in TYPE_MAPPING:
                return TYPE_MAPPING[value]
            # Try to evaluate as a type expression
            try:
                return eval(value, search_frame(value))  # pylint: disable=eval-used
            except Exception as e:
                raise ValueError(f"Invalid type: {value}") from e
        if isinstance(value, (type, GenericAlias, _GenericAlias, _UnionGenericAlias)):
            TypeAdapter(value).rebuild(force=True)
            return value
        raise ValueError(f"Expected type or type name, got {type(value)}")

    @classmethod
    def serialize(cls, value: Type) -> str:
        """Serialize type to string representation"""
        if value is NoneType:
            return "NoneType"
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
            "examples": ["str", "int", "list[str]", "TaskDef", "Union[str, int]"],
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
                    core_schema.dict_schema(),
                    core_schema.list_schema(),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(cls.serialize),
        )


class AgentParams(BaseModel):
    result_type: Optional[Annotated[Type, TypeField]] = Field(default=str)
    deps_type: Optional[Annotated[Type, TypeField]] = Field(default_factory=lambda: NoneType)
    model_settings: Optional[Dict] = Field(default=None)
    retries: Optional[int] = Field(default=1)
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

    # def dict(self):
    #     return ResultData


class AgentDef(BaseModel):
    model: Union[config.known_models, Model]
    name: str
    system_prompt: str
    params: Optional[AgentParams] = Field(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
