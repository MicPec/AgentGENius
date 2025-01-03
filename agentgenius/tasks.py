import logging
from pathlib import Path
from typing import Callable, List, Optional, Union

from pydantic import BaseModel, Field
import bisect


class TaskSchema(BaseModel):
    name: str = Field(..., description="The name of the task")
    description: str = Field(..., description="The description of the task")
    priority: int = Field(
        default=0, ge=0, le=10, description="The priority of the task, with 0 being the highest priority"
    )
    agent_name: str = Field(..., description="The name of the agent to use for the task")
    toolset: List[str] = Field(..., description="The list of tool names in the toolset to use for the task")
    done: bool = Field(default=False, description="Whether the task is done")
    result: Optional[str] = Field(None, description="The result of the task")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "my_task",
                    "description": "My task description",
                    "priority": 5,
                    "agent_name": "my_agent",
                    "toolset": ["ask_agent_tool", "search_tool"],
                }
            ]
        }
    }

    def __lt__(self, other):
        return self.priority < other.priority


class TaskQueue(BaseModel):
    tasks: List[TaskSchema] = Field(default_factory=list)

    def get_all(self) -> List[TaskSchema]:
        return sorted(self.tasks)

    def add(self, task: TaskSchema):
        bisect.insort(self.tasks, task)

    def remove(self, task: TaskSchema):
        self.tasks.remove(task)
