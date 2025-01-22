from typing import Tuple

import logfire
from dotenv import load_dotenv
from rich import print

from agentgenius.aggregator import Aggregator
from agentgenius.builtin_tools import *
from agentgenius.history import HistoryItem, TaskHistory, TaskItem
from agentgenius.task_managment import QuestionAnalyzer, TaskRunner
from agentgenius.tasks import TaskDef
from agentgenius.tools_management import ToolManager
from agentgenius.utils import save_task_history, save_final_result

logfire.configure(send_to_logfire="if-token-present")

load_dotenv()


class AgentGENius:
    def __init__(self, model: str = "openai:gpt-4o", max_history: int = 10):
        """
        Initialize the AgentGENius.

        Args:
            model: The model identifier to use for the agent
            max_history: Maximum number of items to keep in task history
        """

        self.model = model
        self.history = TaskHistory(max_items=max_history)

    # Main public interface methods
    async def ask(self, query: str) -> str:
        """Process a query asynchronously."""
        self._store_query(query)
        tasks, task_history = await self._analyze_query(query)
        await self._process_all_tasks(tasks, task_history)
        result = await self._get_final_result(task_history)
        self._update_history(result)
        return result

    def ask_sync(self, query: str) -> str:
        """Process a query synchronously."""
        self._store_query(query)
        tasks, task_history = self._analyze_query_sync(query)
        self._process_all_tasks_sync(tasks, task_history)
        result = self._get_final_result_sync(task_history)
        print(f"Final result: {result}")  # Debug
        self._update_history(result)
        if current_item := self.history.get_current_item():
            print(f"History item after update: {current_item}")  # Debug
        return result

    # Query analysis methods
    async def _analyze_query(self, query: str) -> Tuple[list[TaskDef], TaskHistory]:
        """Analyze the query and break it down into tasks asynchronously."""
        analyzer = QuestionAnalyzer(model=self.model)
        tasks = await analyzer.analyze(query, deps=self.history)
        return tasks, self._create_task_history(query)

    def _analyze_query_sync(self, query: str) -> Tuple[list[TaskDef], TaskHistory]:
        """Synchronous version of query analysis."""
        analyzer = QuestionAnalyzer(model=self.model)
        tasks = analyzer.analyze_sync(query, deps=self.history)
        return tasks, self._create_task_history(query)

    # Task processing methods
    async def _process_all_tasks(self, tasks: list[TaskDef], task_history: TaskHistory) -> None:
        """Process all tasks asynchronously."""
        for task_def in tasks:
            await self._process_single_task(task_def, task_history)

    def _process_all_tasks_sync(self, tasks: list[TaskDef], task_history: TaskHistory) -> None:
        """Process all tasks synchronously."""
        for task_def in tasks:
            self._process_single_task_sync(task_def, task_history)

    @save_task_history()
    async def _process_single_task(self, task_def: TaskDef, task_history: TaskHistory) -> None:
        """Process a single task asynchronously."""
        tool_manager = ToolManager(model=self.model, task_def=task_def)
        tools = await tool_manager.analyze()
        task = TaskRunner(model=self.model, task_def=task_def, toolset=tools)
        result = await task.run(task_history)
        self._store_task_result(task_def, result.data)
        if current_item := task_history.get_current_item():
            current_item.tasks.append(TaskItem(name=task_def.name, result=result.data))

    @save_task_history()
    def _process_single_task_sync(self, task_def: TaskDef, task_history: TaskHistory) -> None:
        """Process a single task synchronously."""
        tool_manager = ToolManager(model=self.model, task_def=task_def)
        tools = tool_manager.analyze_sync()
        task = TaskRunner(model=self.model, task_def=task_def, toolset=tools)
        result = task.run_sync(task_history)
        self._store_task_result(task_def, result.data)
        if current_item := task_history.get_current_item():
            current_item.tasks.append(TaskItem(name=task_def.name, result=result.data))

    # Result aggregation methods
    async def _get_final_result(self, task_history: TaskHistory) -> str:
        """Aggregate results from all tasks asynchronously."""
        aggregator = Aggregator(model=self.model, history=task_history)
        result = await aggregator.analyze()
        print(f"Aggregator result: {result}")  # Debug
        return result

    def _get_final_result_sync(self, task_history: TaskHistory) -> str:
        """Aggregate results from all tasks synchronously."""
        aggregator = Aggregator(model=self.model, history=task_history)
        result = aggregator.analyze_sync()
        print(f"Aggregator result: {result}")  # Debug
        return result

    # History management methods
    def _store_query(self, query: str) -> None:
        """Store the initial query in history."""
        self.history.append(HistoryItem(user_query=query))

    def _create_task_history(self, query: str) -> TaskHistory:
        """Create a new task history for the current query."""
        task_history = TaskHistory()
        task_history.append(HistoryItem(user_query=query))
        return task_history

    @save_final_result()
    def _update_history(self, result: str) -> None:
        """Update the main history with the final result."""
        if current_item := self.history.get_current_item():
            print(f"Updating history with result: {result}")  # Debug
            current_item.result = result
            print(f"History item after update: {current_item}")  # Debug

    def _store_task_result(self, task_def: TaskDef, result: str) -> None:
        """Store a task result in the current history item."""
        if current_item := self.history.get_current_item():
            current_item.tasks.append(TaskItem(name=task_def.name, result=result))


if __name__ == "__main__":
    agentgenius = AgentGENius()
    print(agentgenius.ask_sync("what time is it?, what is the weather now?"))
    print(agentgenius.history)
