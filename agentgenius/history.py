from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class TaskItem(BaseModel):
    """Individual task item in history"""

    name: str
    result: str
    # timestamp: datetime = Field(default_factory=datetime.now)


class HistoryItem(BaseModel):
    """Single history item containing user query, tasks and final result"""

    user_query: str
    tasks: List[TaskItem] = Field(default_factory=list)
    result: Optional[str] = None
    # timestamp: datetime = Field(default_factory=datetime.now)


class TaskHistory(BaseModel):
    """Task history container with max items limit"""

    max_items: int = Field(default=10, description="Maximum number of history items to keep")
    items: List[HistoryItem] = Field(default_factory=list, description="History items")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> HistoryItem:
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

    def __str__(self):
        return str(self.items)

    def append(self, item: HistoryItem) -> None:
        """Add new item to history, removing oldest if max_items limit is reached"""
        self.items.append(item)  # pylint: disable=no-member
        if len(self) > self.max_items:
            self.items.pop(0)  # pylint: disable=no-member

    def get_current_item(self) -> Optional[HistoryItem]:
        """Get the most recent history item"""
        return self.items[-1] if self.items else None

    def add_task(self, name: str, result: str) -> None:
        """Add a task to the current history item"""
        if current_item := self.get_current_item():
            current_item.tasks.append(TaskItem(name=name, result=result))

    def set_result(self, result: str) -> None:
        """Set the final result for the current history item"""
        if current_item := self.get_current_item():
            current_item.result = result
