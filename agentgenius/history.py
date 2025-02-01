from typing import List, Optional

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    tool: str = Field(..., description="Tool name")
    args: str = Field(..., description="Tool arguments")
    result: str = Field(..., description="Tool result")


class TaskItem(BaseModel):
    """Individual task item containing query and result"""

    query: str = Field(..., description="Task query")
    tool_results: Optional[List[ToolResult]] = Field(default_factory=list, description="Tool results")
    result: str = Field(..., description="Task result")


class TaskHistory(BaseModel):
    """Single history item containing user query, tasks and final result"""

    user_query: str = Field(..., description="User query")
    tasks: List[TaskItem] = Field(default_factory=list, description="Task list")
    final_result: Optional[str] = Field(default=None, description="Final result")


class History(BaseModel):
    """Container for task history items"""

    max_items: int = Field(default=10, description="Maximum number of history items to keep")
    items: List[TaskHistory] = Field(default_factory=list, description="History items")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> TaskHistory:
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

    def __str__(self):
        return str(self.items)

    def append(self, item: TaskHistory) -> None:
        """Add new item to history, removing oldest if max_items limit is reached"""
        self.items.append(item)  # pylint: disable=no-member
        if len(self) > self.max_items:
            self.items.pop(0)  # pylint: disable=no-member

    def get_current_item(self) -> Optional[TaskHistory]:
        """Get the most recent history item"""
        return self.items[-1] if self.items else None

    def add_task(self, query: str, result: str) -> None:
        """Add a task to the current history item"""
        if current_item := self.get_current_item():
            current_item.tasks.append(TaskItem(query=query, result=result))

    def set_final_result(self, result: str) -> None:
        """Set the final result for the current history item"""
        if current_item := self.get_current_item():
            current_item.final_result = result
