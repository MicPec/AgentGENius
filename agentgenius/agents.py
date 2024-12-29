from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool

from .tools import ToolSet


class AgentSchema(BaseModel):
    """Schema for Agent serialization/deserialization"""

    model: str = Field(..., description="The model identifier")
    system_prompt: str = Field(..., description="The system prompt for the agent")
    toolset: Optional[List[str]] = Field(None, description="List of tool names in the toolset")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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

    def __init__(self, model: str, system_prompt: str, toolset: ToolSet | None = None):
        # super().__init__()
        self.model = model
        self.agent = Agent(model=model, system_prompt=system_prompt)
        self.toolset = toolset
        if self.toolset:
            for tool in self.toolset:
                self.agent.tool(tool, retries=3)
                # self.agent._register_function(tool, takes_ctx=True, retries=3, prepare=None)

    def to_dict(self):
        return AgentSchema(
            model=self.model,
            system_prompt=self.get_system_prompt(),
            toolset=self.toolset.to_dict() if self.toolset else None,
        ).model_dump()

    def to_json(self) -> str:
        return AgentSchema(
            model=self.model,
            system_prompt=self.get_system_prompt(),
            toolset=self.toolset.to_dict() if self.toolset else None,
        ).model_dump_json()

    @property
    def system_prompt(self) -> callable:
        return self.agent.system_prompt

    def get_system_prompt(self) -> str:
        return self.agent._system_prompts[0]

    def run_sync(self, query: str, **kwargs) -> str:
        return self.agent.run_sync(query, **kwargs)

    async def run(self, query: str, **kwargs) -> str:
        return await self.agent.run(query, **kwargs)
