from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, Tool

from agentgenius.agents import AgentDef
from agentgenius.tools import ToolDef, ToolSet


class TaskDef(BaseModel):
    """A task definition with associated agent and toolset.

    Attributes:
        name (str): The name of the task.
        query (str): The question or command to ask the agent.
        priority (int): The priority of the task, lower values are executed first.
        agent_def (Optional[AgentDef]): The agent to use for running the task, optional.
        toolset (Optional[ToolSet]): The toolset to use for running the task, optional.

    Note:
        The `agent_def` and `toolset` attributes are optional and can be omitted and set during task creation.
        However, the `agent_def` attribute is required in the Task class.
    """

    name: str
    query: str = Field(..., description="The question or command to ask the agent.")
    priority: int = Field(default=1, ge=1, le=10)
    agent_def: Optional[AgentDef] = Field(default=None)
    toolset: Optional[ToolSet] = Field(default_factory=ToolSet)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __lt__(self, other):
        return self.priority < other.priority


class TaskStatus(BaseModel):
    """Status update for a task execution.

    Attributes:
        task_name: Name of the current task
        status: Current status message
        progress: Optional progress value (0-100)
    """

    task_name: str
    status: str
    progress: Optional[float] = None


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
    callback: Optional[Callable[[TaskStatus], None]] = Field(default=None, exclude=True, description="Status callback")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __init__(self, **data):
        task_def = data["task_def"]
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

        # merge toolsets
        t1 = (
            task_def.toolset
            if isinstance(task_def, TaskDef)
            else task_def.get("toolset")
            if isinstance(task_def, dict)
            else None
        )
        t1 = t1 if t1 is not None else ToolSet()
        t2 = data["toolset"] if "toolset" in data and data["toolset"] is not None else ToolSet()
        toolset = t1 | t2

        callback = data["callback"] if "callback" in data else None

        super().__init__(task_def=task_def, agent_def=agent_def, toolset=toolset, callback=callback)

        self._agent = Agent(
            model=self.agent_def.model,  # pylint: disable=no-member
            name=self.agent_def.name,  # pylint: disable=no-member
            system_prompt=self.agent_def.system_prompt,  # pylint: disable=no-member
            tools=self._prepare_tools(self.toolset) if self.toolset is not None else [],
            **self.agent_def.params.__dict__ if self.agent_def.params is not None else {},  # pylint: disable=no-member
        )

    def rebuild(self):
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
            try:
                if hasattr(tool, "function"):
                    result.append(Tool(tool.function))
                elif callable(tool):
                    result.append(Tool(tool))
            except Exception as e:
                print(f"Failed to prepare tool {tool}: {str(e)}")
        return result

    def _emit_status(self, status: str, progress: Optional[float] = None):
        if self.callback:
            self.callback(TaskStatus(task_name=self.task_def.name, status=status, progress=progress))  # pylint: disable=no-member

    def register_tool(self, tool: ToolDef) -> bool:
        """Registers a tool to the task's agent dynamically."""
        try:
            if hasattr(tool, "function"):
                self.agent._register_tool(Tool(tool.function))  # pylint: disable=protected-access
            elif callable(tool):
                self.agent._register_tool(Tool(tool))  # pylint: disable=protected-access
            self.toolset.add(tool)  # pylint: disable=no-member
            return True
        except Exception as e:
            print(f"Failed to register tool {tool}: {str(e)}")
            return False

    def register_toolset(self, toolset: ToolSet):
        """Registers toolset to the task's agent dynamically."""
        if not toolset:
            return
        for tool in toolset:
            self.register_tool(tool)

    async def run(self, *args, **kwargs):
        query = self.task_def.query  # pylint: disable=no-member
        if self.task_def.query and args:  # pylint: disable=no-member
            query = f"{self.task_def.query}: {args[0]}"  # pylint: disable=no-member
        self._emit_status("Running task", None)
        try:
            result = await self.agent.run(query, **kwargs)
            self._emit_status("Task completed", 100)
            return result
        except Exception as e:
            self._emit_status(f"Task failed: {str(e)}", None)
            raise e

    def run_sync(self, *args, **kwargs):
        query = self.task_def.query  # pylint: disable=no-member
        if self.task_def.query and args:  # pylint: disable=no-member
            query = f"{self.task_def.query}: {args[0]}"  # pylint: disable=no-member
        self._emit_status("Running task", None)
        try:
            result = self.agent.run_sync(query, **kwargs)
            self._emit_status("Task completed", 100)
            return result
        except Exception as e:
            self._emit_status(f"Task failed: {str(e)}", None)
            raise e

    @property
    def agent(self):
        return self._agent


class TaskList(BaseModel):
    tasks: list[TaskDef] = Field(default_factory=list)

    def __iter__(self):
        return iter(self.tasks)

    def __getitem__(self, item):
        return self.tasks[item]

    def append(self, task: TaskDef):
        self.tasks.append(task)  # pylint: disable=no-member

    def sorted(self):
        return sorted(self.tasks)
