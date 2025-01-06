from copy import deepcopy
from dataclasses import field
from typing import Callable, List, Sequence, Union

from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass
from pydantic_ai.tools import Tool


@dataclass(init=False)
class ToolDef:
    """
    Represents a tool definition that encapsulates a callable function identified by its name.
    The tool can be invoked directly and is dynamically resolved from the provided namespace.

    Args:
        name: A string representing the name of the tool, used to retrieve the corresponding callable.
    """

    name: str = field(default="", repr=True)
    # _function: Optional[Tool] = field(default=None, repr=False, init=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __post_init__(self):
        self._function = self._get_callable(namespace=globals())
        self.__qualname__ = f"ToolDef.{self.name}"

    def _get_func(self):
        return getattr(self, "_function", None)

    def _get_callable(self, *, namespace: dict) -> Callable:
        namespace = globals() | namespace
        result = namespace.get(self.name, None)
        if result:
            self._function = result
            return result
        else:
            raise ValueError(f"Tool {self.name} not found in namespace")

    def __call__(self):
        return self._function()

    @property
    def __name__(self) -> str:
        return self.name


ToolType = Union[str, Callable, Sequence[Union[str, Callable]]]


@dataclass(init=False)
class ToolSet:
    """
    A collection of ToolDef objects, representing tools that can be dynamically resolved and invoked.

    Args:
        tools (List[ToolDef]): A list of ToolDef objects encapsulating callable functions.
                Can be a list of strings, callables, or a combination of both.

    Methods:
        add(tool): Adds a tool to the ToolSet.
        remove(name): Removes a tool by name from the ToolSet.
        get(name, default, *, namespace): Retrieves a tool by name from the ToolSet.
        all(): Returns a list of all tool names in the ToolSet.
        init(*, namespace): Initializes tools in the ToolSet with a given namespace.
    """

    tools: List[ToolDef] = field(init=True, default_factory=list, repr=True)

    @field_validator("tools", mode="plain")
    @classmethod
    def accept_other(cls, v):
        if isinstance(v, (list, str, Callable)):
            return v
        else:
            raise ValueError(f"Tool must be a callable or a string, not {type(v)}")

    def __post_init__(self):
        tools = deepcopy(self.tools)
        self.tools = []
        for tool in tools:
            self.add(tool)

    def add(self, tool: ToolType) -> None:
        if isinstance(tool, Callable):
            t = ToolDef(name=tool.__name__)
            self.tools.append(t)
        elif isinstance(tool, str):
            t = ToolDef(name=tool)
            self.tools.append(t)
        elif isinstance(tool, list):
            for t in tool:
                self.add(t)
        else:
            raise ValueError(f"Tool must be a callable or a string, not {type(tool)}")

    def remove(self, name: str) -> Tool:
        tool = self.get(name)
        if tool:
            self.tools.remove(tool)
        return tool

    def get(self, name: str, default=None, *, namespace: dict = globals()):
        return next((tool._get_callable(namespace=namespace) for tool in self.tools if tool.name == name), default)

    def __iter__(self):
        return iter(self.tools)

    def __getitem__(self, item):
        return self.tools[item]._get_callable(namespace=globals())

    def __len__(self):
        return len(self.tools)

    def __contains__(self, value):
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

    # TODO: fix namespace injection
    def init(self, *, namespace: dict = globals()):
        namespace = globals() | namespace
        # return [(tool._get_callable(namespace=namespace).__name__) for tool in self.tools]
        for tool in self.tools:
            tool._get_callable(namespace=namespace)
