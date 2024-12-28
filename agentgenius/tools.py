from pydantic import BaseModel, Field
from pydantic_ai import Tool
from typing import Union, Callable
from dataclasses import dataclass, field
# from collections.abc import Callable


@dataclass
class ToolSet:
    _tools: list[Callable] = field(default_factory=list)

    def __init__(self, tools: Union[list[Callable], Callable, None] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tools = []
        if tools:
            if isinstance(tools, Callable):
                self.add(tools)
            elif isinstance(tools, list):
                for tool in tools:
                    self.add(tool)
            else:
                raise ValueError("Tools must be a list of callable tools or a single callable tool")

    def __iter__(self):
        return iter(self._tools)

    def __len__(self):
        return len(self._tools)

    def __getitem__(self, index):
        return self._tools[index]

    def __setitem__(self, index, value):
        self._tools[index] = value

    def __delitem__(self, index):
        del self._tools[index]

    def __contains__(self, item):
        return item in self._tools

    def __all__(self):
        return self._tools

    def get(self, name, default=None):
        return next((tool for tool in self._tools if tool.name == name), default)

    def add(self, tool: Tool):
        self._tools.append(tool)

    def remove(self, tool: Tool):
        self._tools.remove(tool)

    # def all(self):
    #     return self._tools
