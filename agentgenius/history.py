from typing import TypeVar

from pydantic import BaseModel, Field

TaskHistoryItem = TypeVar("TaskHistoryItem", bound=dict)


class TaskHistory(BaseModel):
    max_items: int = 10
    items: list[TaskHistoryItem] = Field(default_factory=list, description="Task history items")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> TaskHistoryItem:
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

    def __str__(self):
        return str(self.items)

    def append(self, item: TaskHistoryItem) -> None:
        self.items.append(item)  # pylint: disable=no-member
        if len(self) > self.max_items:
            self.items.pop(0)  # pylint: disable=no-member
