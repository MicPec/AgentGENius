from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)
from pydantic.dataclasses import dataclass
from pydantic_ai import Agent, Tool

from agentgenius import tools
from agentgenius.agents import AgentDef
from agentgenius.tools import ToolDef, ToolSet


class TaskDef(BaseModel):
    """A task definition with associated agent and toolset.

    Attributes:
        name (str): The name of the task.
        question (str): The question to ask the agent.
        priority (int): The priority of the task, lower values are executed first.
        agent (Optional[AgentDef]): The agent to use for running the task, optional.
        toolset (Optional[ToolSet]): The toolset to use for running the task, optional.

    Note:
        The `agent` and `toolset` attributes are optional and can be omitted and set during task creation.
        However, the `agent_def` and `toolset` attributes are required in the Task class.
    """

    name: str
    question: str
    priority: int = Field(default=1, ge=1, le=10)
    agent_def: Optional[AgentDef] = Field(default=None)
    toolset: Optional[ToolSet] = Field(default_factory=ToolSet)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # json_encoders={type: custom_type_encoder},
        # defer_build=True,
    )

    def __lt__(self, other):
        return self.priority < other.priority


class Task(BaseModel):
    """
    A task with an associated agent and toolset.

    Attributes:
        agent (Agent): The agent to use for running the task.
        task (TaskDef): The task definition.
        agent_def (AgentDef): The agent definition. Can be omitted, but task_def.agent_def is required in this case.
        toolset (ToolSet): The toolset to use for running the task. Defaults to an empty list.
    """

    task_def: TaskDef = Field(default_factory=TaskDef)
    agent_def: Optional[AgentDef] = Field(default=None)
    toolset: Optional[ToolSet] = Field(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    # def __init__(self, task_def: TaskDef, agent_def: AgentDef, toolset: ToolSet):  # pylint: disable=redefined-outer-name
    def __init__(self, **data):
        task_def = data["task_def"]
        print(f"{task_def=}")
        print(f"{type(task_def)=}")
        agent_def = (
            data["agent_def"]
            if "agent_def" in data and data["agent_def"] is not None
            else task_def.agent_def
            if isinstance(task_def, TaskDef)
            else task_def["agent_def"]
            if "agent_def" in task_def
            else None
        )
        # task_def.agent_def = agent_def
        if agent_def is None:
            raise ValueError("AgentDef is required ether in constructor or in TaskDef")
        toolset = (
            data["toolset"]
            if "toolset" in data and data["toolset"] is not None
            else task_def.toolset
            if isinstance(task_def, TaskDef)
            else task_def["toolset"]
            if "toolset" in task_def
            else None
        )
        super().__init__(task_def=task_def, agent_def=agent_def, toolset=toolset)

        self._agent = Agent(
            model=self.agent_def.model,  # pylint: disable=no-member
            name=self.agent_def.name,  # pylint: disable=no-member
            system_prompt=self.agent_def.system_prompt,  # pylint: disable=no-member
            tools=self._prepare_tools(self.toolset) if self.toolset is not None else [],
            **self.agent_def.params.__dict__ if self.agent_def.params is not None else {},  # pylint: disable=no-member
        )

    def _prepare_tools(self, t: list) -> list[Tool]:
        result = []
        for tool in t:
            result.append(Tool(tool.function))
        return result

    def register_tool(self, tool: ToolDef):
        """Registers a tool to the task's agent dynamically."""
        self._agent._register_tool(Tool(tool.function))  # pylint: disable=protected-access

    def register_toolset(self, toolset: ToolSet):
        """Registers toolset to the task's agent dynamically."""
        for tool in toolset:
            self.register_tool(tool)

    async def run(self, *args, **kwargs):
        question = self.task_def.question  # pylint: disable=no-member
        if self.task_def.question and args:  # pylint: disable=no-member
            question = f"{self.task_def.question}: {args[0]}"  # pylint: disable=no-member
        return await self._agent.run(question, **kwargs)

    def run_sync(self, *args, **kwargs):
        question = self.task_def.question  # pylint: disable=no-member
        if self.task_def.question and args:  # pylint: disable=no-member
            question = f"{self.task_def.question}: {args[0]}"  # pylint: disable=no-member
        return self._agent.run_sync(question, **kwargs)


class TaskList(BaseModel):
    tasks: list[TaskDef] = Field(default_factory=list)

    def __iter__(self):
        return iter(self.tasks)

    def __getitem__(self, item):
        return self.tasks[item]

    def append(self, task: TaskDef):
        self.tasks.append(task)


# @classmethod
# def __get_pydantic_core_schema__(cls, source: any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
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
