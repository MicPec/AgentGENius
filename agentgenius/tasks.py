from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
)
from pydantic.dataclasses import dataclass
from pydantic_ai import Agent, Tool

from agentgenius import tools
from agentgenius.agents import AgentDef
from agentgenius.tools import ToolDef, ToolSet
from agentgenius.utils import TypeAdapterMixin, custom_type_encoder


# @dataclass(init=False)
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
    """

    name: str
    question: str
    priority: int = Field(default=1, ge=1, le=10)
    agent_def: Optional[AgentDef] = Field(default=None)
    toolset: Optional[ToolSet] = Field(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={type: custom_type_encoder},
        # defer_build=True,
    )

    @field_validator("agent_def")
    @classmethod
    def validate_agent(cls, value):
        if isinstance(value, dict):
            value = AgentDef.model_validate(value)
        return value

    @field_validator("toolset")
    @classmethod
    def validate_toolset(cls, value):
        if isinstance(value, dict):
            value = ToolSet.model_validate(value)
        return value

    def __lt__(self, other):
        return self.priority < other.priority

    @field_serializer("agent_def")
    def _serialize_agent_def(self, value: AgentDef):
        return value.model_dump() if value else None

    @field_serializer("toolset")
    def _serialize_toolset(self, value: ToolSet):
        return value.model_dump() if value else None


# @dataclass
class Task(BaseModel):
    """
    A task with an associated agent and toolset.

    Attributes:
        agent (Agent): The agent to use for running the task.
        task (TaskDef): The task definition.
        agent_def (AgentDef): The agent definition.
        toolset (ToolSet): The toolset to use for running the task. Defaults to an empty list.
    """

    task_def: TaskDef = Field(default_factory=TaskDef)
    agent_def: AgentDef = Field(default=None)
    toolset: ToolSet = Field(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={type: custom_type_encoder},
        # defer_build=True,
    )

    @field_validator("agent_def")
    @classmethod
    def validate_agent_def(cls, value):
        if isinstance(value, dict):
            value = AgentDef.model_validate(value)
        return value

    def __init__(self, task_def: TaskDef, agent_def: AgentDef = None, toolset: ToolSet = None):  # pylint: disable=redefined-outer-name
        agent_def = agent_def if agent_def is not None else task_def.agent_def
        toolset = toolset if toolset is not None else task_def.toolset
        super().__init__(task_def=task_def, agent_def=agent_def, toolset=toolset)
        self._agent = Agent(
            model=self.agent_def.model,  # pylint: disable=no-member
            name=self.agent_def.name,  # pylint: disable=no-member
            system_prompt=self.agent_def.system_prompt,  # pylint: disable=no-member
            tools=self._prepare_tools(self.toolset),
            **self.agent_def.params.model_dump(exclude_unset=True) if self.agent_def.params else {},  # pylint: disable=no-member
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


@dataclass(init=False)
class TaskList:
    tasks: list[TaskDef]


#     def __iter__(self):
#         return iter(self.tasks)

#     def __getitem__(self, item):
#         return self.tasks[item]

#     def append(self, task: TaskDef):
#         self.tasks.append(task)


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
