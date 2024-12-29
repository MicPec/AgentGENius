from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool

from .tools import ToolSet
from pathlib import Path


class AgentSchema(BaseModel):
    """Schema for Agent serialization/deserialization"""

    name: str = Field(..., description="The name of the agent")
    model: str = Field(..., description="The model identifier")
    system_prompt: str = Field(..., description="The system prompt for the agent")
    toolset: Optional[List[str]] = Field(None, description="List of tool names in the toolset")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "my_agent",
                    "model": "openai:gpt-4o-mini",
                    "system_prompt": "You are a helpful assistant.",
                    "toolset": ["ask_user_tool"],
                }
            ]
        }
    }


# @dataclass
class BaseAgent:
    # agent: Agent = field(default=None)
    # model: str = field(default="openai:gpt-4o-mini")
    # toolset: ToolSet | None = field(default=None)

    @classmethod
    def from_json(cls, json_str: str, namespace=None):
        data = AgentSchema.model_validate_json(json_str).model_dump()
        if data.get("toolset"):
            data["toolset"] = ToolSet.from_dict(data["toolset"], namespace=namespace)
        return cls(**data)

    @classmethod
    def from_file(cls, file_path: Path, namespace=None):
        with open(file_path, "r") as f:
            return cls.from_json(f.read(), namespace=namespace)

    def __init__(self, name: str, model: str, system_prompt: str, toolset: ToolSet | None = None, **kwargs):
        # super().__init__()
        # self.name = name
        self.model = model
        self.agent = Agent(model=model, system_prompt=system_prompt, name=name, **kwargs)
        self.toolset = toolset
        if self.toolset:
            for tool in self.toolset:
                self.agent.tool(tool, retries=3)
                # self.agent._register_function(tool, takes_ctx=True, retries=3, prepare=None)

    def to_dict(self):
        return AgentSchema(
            name=self.name,
            model=self.model,
            system_prompt=self.get_system_prompt(),
            toolset=self.toolset.to_dict() if self.toolset else None,
        ).model_dump()

    def to_json(self) -> str:
        return AgentSchema(
            name=self.name,
            model=self.model,
            system_prompt=self.get_system_prompt(),
            toolset=self.toolset.to_dict() if self.toolset else None,
        ).model_dump_json()

    @property
    def system_prompt(self) -> callable:
        return self.agent.system_prompt

    @property
    def tool(self, *args, **kwargs) -> callable:
        return self.agent.tool(*args, **kwargs)

    @property
    def tool_plain(self, *args, **kwargs) -> callable:
        return self.agent.tool_plain(*args, **kwargs)

    @property
    def result_validator(self, *args, **kwargs) -> callable:
        return self.agent.result_validator(*args, **kwargs)

    @property
    def name(self) -> str:
        return self.agent.name

    def get_system_prompt(self) -> str:
        return self.agent._system_prompts[0] if self.agent._system_prompts else ""  # pylint: disable=protected-access

    def run_sync(self, *args, **kwargs) -> str:
        return self.agent.run_sync(*args, **kwargs)

    async def run(self, *args, **kwargs) -> str:
        return await self.agent.run(*args, **kwargs)
