import logging
from pathlib import Path
from typing import Callable, List, Optional, Union

from pydantic import BaseModel, Field

from .config import config

logger = logging.getLogger(__name__)


class ToolSchema(BaseModel):
    """Schema for Tool serialization/deserialization"""

    name: str = Field(..., description="Name of the tool (function name)")
    description: Optional[str] = Field(None, description="Tool description")
    args: Optional[dict] = Field(None, description="Tool arguments and types")
    return_type: Optional[str] = Field(None, description="Tool return type")
    code: Optional[str] = Field(None, description="Tool code")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "get_datetime",
                    "description": "Get current datetime",
                    "args": {"format": "str"},
                    "return_type": "str",
                    "code": "def get_datetime(ctx: RunContext[str], format: str = '%Y-%m-%d %H:%M:%S') -> str:\n    import datetime\n    return datetime.datetime.now().strftime(format)",
                }
            ]
        }
    }


class ToolSetSchema(BaseModel):
    """Schema for ToolSet serialization/deserialization"""

    toolset: List[str] = Field(default_factory=list, description="List of tool names in the toolset")

    model_config = {"json_schema_extra": {"examples": [{"tools": ["ask_user_tool", "search_tool"]}]}}


class ToolSet:
    """A set of tools that can be passed to an agent"""

    @staticmethod
    def tool_from_file(toolname: str, path: Path = config.tools_path):
        def run_func(func_str: str) -> Union[callable, str]:
            namespace = {}
            try:
                logger.info("Executing tool: %s", {toolname})
                exec(func_str, globals(), namespace)
            except Exception as e:
                logger.error("Error executing code: %s", str(e))
                return f"Error executing code: {e}"
            logger.info("Successfully executed tool: %s", toolname)
            return next(f for f in namespace.values() if callable(f))

        with open(path / f"{toolname}.py", "r") as f:
            logger.info("Reading tool from file: %s.py", toolname)
            return run_func(f.read())

    @classmethod
    def from_dict(cls, data: Union[List[str], Callable], namespace) -> "ToolSet":
        namespace = globals() | namespace

        logging.info("ToolSet.from_dict: Attempting to create toolset from data: %s", data)
        logging.debug("Available functions in namespace: %s", [k for k in namespace.keys() if callable(namespace[k])])

        if isinstance(data, list):
            toolset = []
            missing_tools = []
            for func_name in data:
                if func_name in namespace:
                    tool = namespace[func_name]
                    if callable(tool):
                        toolset.append(tool)
                        logger.info("ToolSet.from_dict: Added tool %s to ToolSet", tool.__name__)
                    else:
                        logger.error("ToolSet.from_dict: Found %s in namespace but it's not callable", func_name)
                        raise ValueError(f"Tool {func_name} found in namespace but it's not callable")
                else:
                    missing_tools.append(func_name)
                    logger.warning("ToolSet.from_dict: Tool %s not found in namespace", func_name)

            if missing_tools:
                try:
                    for func_name in missing_tools:
                        tool = ToolSet.tool_from_file(func_name)
                        toolset.append(tool)
                        logger.info("ToolSet.from_dict: Added tool %s to ToolSet", tool.__name__)
                except Exception as e:
                    logger.error("ToolSet.from_dict: Failed to create tool from file: %s", str(e))
                # raise ValueError(f"The following tools were not found in the namespace: {missing_tools}")

            return cls(toolset)
        return cls(data)

    @classmethod
    def from_json(cls, json_str: str, namespace) -> "ToolSet":
        data = ToolSetSchema.model_validate_json(json_str)
        return cls.from_dict(data.toolset, namespace=namespace)

    def __init__(self, tools: Union[list[Callable], Callable, None] = None):
        self.toolset = []
        if tools:
            if isinstance(tools, Callable):
                self.add(tools)
            elif isinstance(tools, list):
                for tool in tools:
                    if tool.__name__ in self._func_names():
                        logging.info("Toolset.__init__: Tool %s already exists in ToolSet", tool.__name__)
                        raise ValueError(f"Tool {tool.__name__} already exists in ToolSet")
                    self.add(tool)
                    logger.info("Toolset.__init__: Added tool %s to ToolSet", tool.__name__)
            else:
                logging.error("Toolset.__init__:Tools must be a list of callable or a single callable.")
                raise ValueError("Tools must be a list of callable or a single callable.")

    def __iter__(self):
        return iter(self.toolset)

    def __contains__(self, value):
        if isinstance(value, str):
            return self.get(value) is not None
        elif isinstance(value, Callable):
            return value in self.toolset
        else:
            raise ValueError("value must be a type of string or callable")

    def __getitem__(self, name):
        return self.get(name)

    def __len__(self):
        return len(self.toolset)

    def __all__(self):
        return self.toolset

    def __repr__(self):
        return f"ToolSet({self._func_names()})"

    def _func_names(self) -> List[str]:
        return [tool.__name__ for tool in self.toolset]

    def get(self, name: str, default=None) -> Optional[Callable]:
        return next((tool for tool in self.toolset if tool.__name__ == name), default)

    def add(self, tool: Callable):
        if tool.__name__ in self._func_names():
            logger.warning("Tool %s already exists in ToolSet", tool.__name__)
            raise ValueError(f"Tool {tool.__name__} already exists in ToolSet")
        self.toolset.append(tool)
        logger.info("Added tool %s to ToolSet", tool.__name__)

    @property
    def list(self):
        return self._func_names()

    def remove(self, tool_name: str):
        if self.get(tool_name) is None:
            raise ValueError(f"Tool {tool_name} not found in ToolSet")
        self.toolset.remove(self.get(tool_name))
        logger.info("Removed tool %s from ToolSet", tool_name)

    def to_dict(self):
        return ToolSetSchema(toolset=self._func_names()).model_dump()["toolset"]

    def to_json(self):
        return ToolSetSchema(toolset=self._func_names()).model_dump_json()
