from pathlib import Path
from typing import List, Optional, overload

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UserError

from .config import config
from .tools import ToolSet


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


class BaseAgent:
    """BaseAgent represents an AI agent with a specific model, system prompt, and toolset.
    This class allows for the creation, serialization, and deserialization of agent instances.
    """

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
        self.agent_store = AgentStore()
        self.model = model
        self.agent = Agent(model=model, system_prompt=system_prompt, name=name, **kwargs)
        self.toolset = toolset if toolset else ToolSet()
        self.toolset.add(self.ask_agent)
        self.system_prompt(self._inject_agents)
        self.system_prompt(self._inject_tools)

    def __repr__(self):
        return f"Agent(name={self.name}, model={self.model}, toolset={self.toolset})"

    def _is_func_plain(self, func: callable) -> bool:
        # return func.__code__.co_argcount == 0
        import inspect

        signature = inspect.signature(func)
        return not any("RunContext" in str(signature.parameters[t].annotation) for t in signature.parameters.keys())

    def _refresh_toolset(self):
        if self.toolset:
            for tool in self.toolset:
                try:
                    if self._is_func_plain(tool):
                        self.agent.tool_plain(tool)
                    else:
                        self.agent.tool(tool)
                except UserError as e:
                    # logger.debug(f"Tool {tool} already exists in Agents toolset")
                    pass

    def _inject_agents(self):
        return f"Agents: {self.agent_store.list()}"

    def _inject_tools(self):
        self._refresh_toolset()
        return f"Tools: {self.toolset}"

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
        # self._refresh_toolset()
        return self.agent.run_sync(*args, **kwargs)

    async def run(self, *args, **kwargs) -> str:
        # self._refresh_toolset()
        return await self.agent.run(*args, **kwargs)

    async def run_stream(self, *args, **kwargs):
        # self._refresh_toolset()
        return await self.agent.run_stream(*args, **kwargs)

    async def ask_agent(self, ctx: RunContext[any], agent_name: str, question: str, deps: list[str]) -> str:
        """Ask a specific agent a question.
        He can solve tasks that you can't solve."""
        if agent := self.agent_store.get(agent_name):
            return await agent.run(question, deps=deps)
        return f"Agent '{agent_name}' not found"

    async def list_agents(self, ctx: RunContext[any]):
        """List all agents that are stored in the agent store.
        Check if any of them can solve the task."""
        print(self.agent_store.list())
        return self.agent_store.list()


class AgentStore:
    """A class to store and manage agents."""

    def __init__(self, path: Path | str = config.agents_path):
        self.agents = {}
        self.path = Path(path)

    def __getitem__(self, name: str) -> BaseAgent:
        if name not in self.agents:
            raise KeyError(f"Agent {name} not found")
        return self.agents[name]

    def list(self) -> List[str]:
        return list(self.agents.keys())

    @overload
    def add(self, agent: BaseAgent, overwrite: bool = False) -> BaseAgent: ...

    @overload
    def add(self, agent: AgentSchema, overwrite: bool = False) -> BaseAgent: ...

    def add(self, agent: BaseAgent | AgentSchema, overwrite: bool = False) -> BaseAgent:
        if isinstance(agent, BaseAgent):
            if overwrite or agent.name not in self.agents:
                self.agents[agent.name] = agent
        elif isinstance(agent, AgentSchema):
            agent = BaseAgent.from_json(agent)
            if overwrite or agent.name not in self.agents:
                self.agents[agent.name] = agent
        else:
            raise ValueError(f"Unsupported agent type: {type(agent)}")
        return self.agents[agent.name]

    def get(self, name: str) -> BaseAgent | None:
        try:
            return self[name]
        except KeyError:
            return None

    def load_agents(self) -> "AgentStore":
        for agent_file in self.path.iterdir():
            if agent_file.name.endswith(".json"):
                agent = BaseAgent.from_file(agent_file)
                self.add(agent)
        return self

    def load_agent(self, name: str) -> BaseAgent | None:
        agent_file = self.path / f"{name}.json"
        if agent_file.exists():
            agent = BaseAgent.from_file(agent_file)
            self.add(agent)
        return agent

    def save_agent(self, agent: BaseAgent) -> str:
        config.agents_path.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member

        result = agent.to_json()
        with open(config.agents_path / f"{agent.name}.json", encoding="utf-8", mode="w") as f:
            f.write(result)
        return result

    def save_agents(self):
        for agent in self.agents.values():
            self.save_agent(agent)
