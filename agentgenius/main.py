from pydantic_ai import RunContext

from .agents import AgentSchema, AgentStore, BaseAgent
from .builtin_tools import (
    get_all_agents_tool,
    get_all_tools_tool,
    get_external_tools_tool,
    get_builtin_tools_tool,
)
from .config import config, prompt_lib
from .tasks import TaskQueue, TaskSchema
from .tools import ToolSchema, ToolSet


class TaskGENius(BaseAgent):
    def __init__(
        self,
        model: str | None = config.default_agent_model,
        system_prompt: str | None = config.default_agent_prompt,
        toolset: ToolSet | None = None,
    ):
        super().__init__("task planner agent", model, system_prompt, toolset, result_type=TaskQueue)
        self.task_queue: TaskQueue | None = None
        self.toolset.add(get_all_tools_tool)
        self.toolset.add(get_all_agents_tool)
        # self.system_prompt(f"All available tools:  {ToolSet.list_all_tools()}")

    # async def run(self, *args, **kwargs):


class ToolGENius(BaseAgent):
    def __init__(
        self,
        model: str | None = config.default_agent_model,
        system_prompt: str | None = config.default_agent_prompt,
        toolset: ToolSet | None = None,
    ):
        super().__init__("tool agent", model, system_prompt, toolset, retries=5)
        self.system_prompt(self._inject_builtin_tools)
        self.toolset.add(get_external_tools_tool)
        self.toolset.add(self.init_tool)

    def _inject_builtin_tools(self) -> str:
        return f"Builtin tools available: {ToolSet.list_builtin_tools()}"

    def _inject_external_tools(self) -> str:
        return f"External tools available: {ToolSet.list_external_tools()}"

    # TODO: fix this
    # async def run(self, task: str, **kwargs):
    #     return await self.agent.run(task, **kwargs)

    def init_tool(self, tool: ToolSchema) -> str:
        """Add a tool to the toolset and return its name.
        Then you can call this tool directly from toolset by its name (returned by this function)."""
        # print(f"{tool=}")
        exec_tool = ToolSet.tool_from_schema(tool)
        self.toolset.add(exec_tool)
        self._refresh_toolset()
        return exec_tool.__name__


class AgentGENius(BaseAgent):
    def __init__(
        self,
        name: str,
        model: str | None = config.default_agent_model,
        system_prompt: str | None = prompt_lib["agentgenius"],
        toolset: ToolSet | None = None,
    ):
        super().__init__(name, model, system_prompt, toolset)
        self.planner_agent = TaskGENius(config.default_agent_model, prompt_lib["planner_agent"])
        self.tool_agent = ToolGENius(config.default_agent_model, prompt_lib["tool_agent"])
        self.agent_store.add(self.planner_agent)
        self.agent_store.add(self.tool_agent)

    def run_system(self, task: str):
        pass
