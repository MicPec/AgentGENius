from dataclasses import field
from typing import Optional

from pydantic import ValidationInfo, field_validator
from pydantic.dataclasses import dataclass
from pydantic_ai import Agent, Tool

from agentgenius import tools
from agentgenius.agents import AgentDef
from agentgenius.tools import ToolDef, ToolSet


@dataclass(init=False)
class TaskDef:
    """A task definition with associated agent and toolset.

    Attributes:
        name (str): The name of the task.
        question (str): The question to ask the agent.
        priority (int): The priority of the task, lower values are executed first.
        agent (Optional[AgentDef]): The agent to use for running the task, optional.
        toolset (Optional[ToolSet]): The toolset to use for running the task, optional.

    Note:
        The `agent` and `toolset` attributes are optional and can be omitted and set during task creation.
    """

    name: str
    question: str
    priority: int
    agent: Optional[AgentDef] = field(default=None, init=True, repr=False)
    toolset: Optional[ToolSet] = field(default=None, init=True, repr=False)

    def __lt__(self, other):
        return self.priority < other.priority


@dataclass(init=False)
class Task:
    """
    A task with an associated agent and toolset.

    Attributes:
        agent (Agent): The agent to use for running the task.
        task (TaskDef): The task definition.
        agent_def (AgentDef): The agent definition.
        toolset (ToolSet): The toolset to use for running the task. Defaults to an empty list.
    """

    agent: Agent = field(init=False, repr=False)
    task_def: TaskDef
    agent_def: AgentDef = None
    toolset: ToolSet = field(default_factory=list, init=True, repr=True)

    def __post_init__(self):
        if self.agent_def is None and self.task_def.agent is not None:
            # if agent_def is not provided, use the agent definition from the task_def
            self.agent_def = self.task_def.agent

        if self.toolset == [] and self.task_def.toolset is not None:
            # if toolset is not provided, use the toolset from the task_def
            self.toolset = self.task_def.toolset

        self.agent = Agent(
            model=self.agent_def.model,
            name=self.agent_def.name,
            system_prompt=self.agent_def.system_prompt,
            # tools=[Tool(tool.function) for tool in self.toolset],
            tools=self._prepare_tools(self.toolset),  # Done this way for debugging
            **self.agent_def.params if self.agent_def.params else {},
        )

    def _prepare_tools(self, t: list) -> list[Tool]:
        result = []
        for tool in t:
            result.append(Tool(tool.function))
        return result

    def register_tool(self, tool: ToolDef):
        """Registers a tool to the task's agent dynamically."""
        self.agent._register_tool(Tool(tool.function))

    def register_toolset(self, toolset: ToolSet):
        """Registers toolset to the task's agent dynamically."""
        for tool in toolset:
            self.register_tool(tool)

    async def run(self, *args, **kwargs):
        question = self.task_def.question
        if self.task_def.question and args:
            question = f"{self.task_def.question}: {args[0]}"
        return await self.agent.run(question, **kwargs)

    def run_sync(self, *args, **kwargs):
        question = self.task_def.question
        if self.task_def.question and args:
            question = f"{self.task_def.question}: {args[0]}"
        return self.agent.run_sync(question, **kwargs)


@dataclass(init=False)
class TaskList:
    tasks: list[TaskDef]

    def __iter__(self):
        return iter(self.tasks)

    def __getitem__(self, item):
        return self.tasks[item]

    def append(self, task: TaskDef):
        self.tasks.append(task)

    # @classmethod
    # def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
    #     instance_schema = core_schema.is_instance_schema(cls)

    #     args = get_args(source)
    #     if args:
    #         # replace the type and rely on Pydantic to generate the right schema
    #         # for `Sequence`
    #         sequence_t_schema = handler.generate_schema(list[args[0]])
    #     else:
    #         sequence_t_schema = handler.generate_schema(list)

    #     non_instance_schema = core_schema.no_info_after_validator_function(TaskList, sequence_t_schema)
    #     return core_schema.union_schema([instance_schema, non_instance_schema])
