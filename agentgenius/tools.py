import json
from dataclasses import dataclass, field
from typing import Callable, Union

# from pydantic import BaseModel, Field


class ToolSet:
    """A set of tools that can be passed to an agent"""

    @classmethod
    def from_dict(cls, data, namespace=None):
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
    def from_json(cls, json_str, namespace=None):
        return cls.from_dict(json.loads(json_str), namespace=namespace)

    def __init__(self, tools: Union[list[Callable], Callable, None] = None):
        # super().__init__(*args, **kwargs)
        self.tools = []
        if tools:
            if isinstance(tools, Callable):
                self.add(tools)
            elif isinstance(tools, list):
                for tool in tools:
                    self.add(tool)
            else:
                raise ValueError("Tools must be a list of callable tools or a single callable tool")

    def __iter__(self):
        return iter(self.tools)

    def __all__(self):
        return self.tools

    def __repr__(self):
        return f"ToolSet({self._func_names()})"

    def __getitem__(self, name):
        return self.get(name)

    def __contains__(self, value):
        if isinstance(value, str):
            return self.get(value) is not None
        elif isinstance(value, Callable):
            return value in self.tools
        else:
            raise ValueError("value must be a type of string or callable")

    def _func_names(self):
        return [tool.__name__ for tool in self.tools]  # [tool.name for tool in self.tools]

    def get(self, name, default=None):
        return next((tool for tool in self.tools if tool.__name__ == name), default)

    def add(self, tool: Callable):
        self.tools.append(tool)

    def remove(self, toolname: str):
        self.tools.remove(self.get(toolname))

    def to_dict(self):
        return self._func_names()

    def to_json(self):
        return json.dumps(self.to_dict())
