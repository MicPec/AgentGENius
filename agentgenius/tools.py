from copy import deepcopy
from dataclasses import field
from typing import Callable, List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass
from pydantic_ai.tools import Tool

from agentgenius.utils import custom_type_encoder, search_frame


class ToolDef(BaseModel):
    """
    Represents a tool definition that encapsulates a callable function identified by its name.
    The tool can be invoked directly and is dynamically resolved from the global namespace.

    Args:
        name: A string representing the name of the tool, used to retrieve the corresponding callable.
    """

    name: str = field(default="", repr=True)
    # code: Optional[str] = field(default=None, repr=True)
    # _function: Optional[Tool] = field(default=None, repr=False, init=False)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={type: custom_type_encoder},
        # defer_build=True,
    )

    def __init__(self, name: str):
        super().__init__(name=name)

        if "functions." in self.name:
            self.name = self.name.split("functions.")[1]
        frame = search_frame(self.name)
        self._function = self._get_callable(namespace=frame)
        # self.__qualname__ = self._function.__qualname__

    @property
    def function(self):
        return self._function

    def _get_callable(self, *, namespace: dict) -> Callable:
        result = namespace.get(self.name, None)
        if result:
            self._function = result
            return result
        else:
            raise ValueError(f"Tool '{self.name}' not found ")

    def __call__(self):
        return self._function()

    @property
    def __name__(self) -> str:
        return self.name


ToolType = Union[str, Callable, Sequence[Union[str, Callable]]]


class ToolSet(BaseModel):
    """
    A collection of ToolDef objects, representing tools that can be dynamically resolved and invoked.

    Args:
        tools (List[ToolDef]): A list of ToolDef objects encapsulating callable functions.
                Can be a list of strings, callables, or a combination of both.

    Methods:
        add(tool): Adds a tool to the ToolSet.
        remove(name): Removes a tool by name from the ToolSet.
        get(name, default): Retrieves a tool by name from the ToolSet.
        all(): Returns a list of all tool names in the ToolSet.
    """

    tools: List[ToolDef] = Field(default_factory=list, kw_only=False, key="tools")

    @field_validator("tools", mode="before")
    @classmethod
    def accept_others(cls, v):
        if isinstance(v, (list, str, dict, Callable)):
            return v
        else:
            raise ValueError(f"Tool must be a callable or a string, not {type(v)}")

    def __init__(self, tools: Union[list[Callable], list[str], dict, Callable, None] = None):
        super().__init__()

        if tools:
            self.tools = []
            # for tool in tools:
            self.add(tools)

    def add(self, tool: ToolType) -> None:
        if isinstance(tool, Callable):
            if not self._check_tool_exists(tool.__name__):
                t = ToolDef(name=tool.__name__)
                self.tools.append(t)  # pylint: disable=no-member
            else:
                raise ValueError(f"Tool {tool.__name__} already exists in ToolSet")
        elif isinstance(tool, str):
            if not self._check_tool_exists(tool):
                t = ToolDef(name=tool)
                self.tools.append(t)  # pylint: disable=no-member
            else:
                raise ValueError(f"Tool {tool} already exists in ToolSet")
        elif isinstance(tool, list):
            for t in tool:
                self.add(t)
        elif isinstance(tool, dict):
            for t in tool.values():
                self.add(t)
        else:
            raise ValueError(f"Tool must be a callable or a string, not {type(tool)}")

    def remove(self, name: str) -> Tool:
        tool = next((tool for tool in self.tools if tool.name == name), None)
        if tool:
            self.tools.remove(tool)  # pylint: disable=no-member
        return tool

    def get(self, name: str, default=None):
        """
        Retrieves a tool by name from the ToolSet
        Returns:
            Callable or default: The tool function if found, otherwise the default value.
        """
        return next((tool.function for tool in self.tools if tool.name == name), default)

    def _check_tool_exists(self, name: str) -> bool:
        return next((True for tool in self.tools if tool.name == name), False)

    def __iter__(self):
        return iter(self.tools)

    def __getitem__(self, item) -> Callable:
        return self.tools[item].function

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, value) -> bool:
        if isinstance(value, str):
            return self.get(value) is not None
        elif isinstance(value, Callable):
            return value in self.tools
        else:
            raise ValueError(f"value must be a type of string or callable, not {type(value)}")

    def __str__(self):
        return str(self.tools)

    def all(self):
        return [tool.name for tool in self.tools]
