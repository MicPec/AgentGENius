from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from .tools import ToolSet
import json


class BaseAgent(Agent):
    toolset: ToolSet | None = field(default=None)

    def __init__(self, toolset: ToolSet = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toolset = toolset
        if self.toolset:
            for tool in self.toolset:
                self.tool(tool, retries=3, prepare=None)
        # self._register_function(self.ask_user_tool, takes_ctx=True, retries=3, prepare=None)

    def to_dict(self):
        return {
            "name": self.name,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "toolset": self.toolset,
        }
