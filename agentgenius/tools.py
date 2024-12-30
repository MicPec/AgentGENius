from typing import Callable, List, Optional, Union

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """Schema for Tool serialization/deserialization"""

    name: str = Field(..., description="Name of the tool (function name)")
    description: Optional[str] = Field(None, description="Tool description")
    args: Optional[dict] = Field(None, description="Tool arguments and types")
    return_type: Optional[str] = Field(None, description="Tool return type")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "ask_user_tool",
                    "description": "Ask the user a question and get their response",
                    "args": {"question": "str"},
                    "return_type": "str",
                }
            ]
        }
    }


class ToolSetSchema(BaseModel):
    """Schema for ToolSet serialization/deserialization"""

    tools: List[str] = Field(default_factory=list, description="List of tool names in the toolset")

    model_config = {"json_schema_extra": {"examples": [{"tools": ["ask_user_tool", "search_tool"]}]}}


class ToolSet:
    """A set of tools that can be passed to an agent"""

    @classmethod
    def from_dict(cls, data: Union[List[str], Callable], namespace=None) -> "ToolSet":
        if namespace is None:
            namespace = globals()

        if isinstance(data, list):
            tools = []
            for func_name in data:
                if func_name in namespace:
                    tool = namespace[func_name]
                    tools.append(tool)
            return cls(tools)
        return cls(data)

    @classmethod
    def from_json(cls, json_str: str, namespace=None) -> "ToolSet":
        data = ToolSetSchema.model_validate_json(json_str)
        return cls.from_dict(data.tools, namespace=namespace)

    def __init__(self, tools: Union[list[Callable], Callable, None] = None):
        self.tools = []
        if tools:
            if isinstance(tools, Callable):
                self.add(tools)
            elif isinstance(tools, list):
                for tool in tools:
                    if tool.__name__ in self._func_names():
                        raise ValueError(f"Tool {tool.__name__} already exists in ToolSet")
                    self.add(tool)
            else:
                raise ValueError("Tools must be a list of callable tools or a single callable tool")

    def __iter__(self):
        return iter(self.tools)

    def __contains__(self, value):
        if isinstance(value, str):
            return self.get(value) is not None
        elif isinstance(value, Callable):
            return value in self.tools
        else:
            raise ValueError("value must be a type of string or callable")

    def __getitem__(self, name):
        return self.get(name)

    def __all__(self):
        return self.tools

    def __repr__(self):
        return f"ToolSet({self._func_names()})"

    def _func_names(self) -> List[str]:
        return [tool.__name__ for tool in self.tools]

    def get(self, name: str, default=None) -> Optional[Callable]:
        return next((tool for tool in self.tools if tool.__name__ == name), default)

    def add(self, tool: Callable):
        if tool.__name__ in self._func_names():
            raise ValueError(f"Tool {tool.__name__} already exists in ToolSet")
        self.tools.append(tool)

    def remove(self, toolname: str):
        if self.get(toolname) is None:
            raise ValueError(f"Tool {toolname} not found in ToolSet")
        self.tools.remove(self.get(toolname))

    def to_dict(self):
        return ToolSetSchema(tools=self._func_names()).model_dump()["tools"]

    def to_json(self):
        return ToolSetSchema(tools=self._func_names()).model_dump_json()
