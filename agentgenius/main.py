from typing import Callable

from dotenv import load_dotenv
from pydantic_ai.models import Model
from rich import print

from agentgenius.aggregator import Aggregator
from agentgenius.config import config
from agentgenius.history import History, TaskHistory, TaskItem
from agentgenius.task_management import QuestionAnalyzer, TaskRunner
from agentgenius.tasks import TaskStatus
from agentgenius.tools_management import ToolManager
from agentgenius.utils import extract_tool_results, save_history

# from agentgenius.tasks import TaskDef

load_dotenv()


class AgentGENius:
    def __init__(
        self,
        model: Model | str = config.default_model,
        max_history: int = 10,
        callback: Callable[[TaskStatus], None] = None,
    ):
        """
        Initialize the AgentGENius.

        Args:
            model: The model identifier to use for the agent
            max_history: Maximum number of items to keep in task history
        """
        self.model = model
        self.callback = callback
        self.history = History(max_items=max_history)

    @save_history()
    async def ask(self, query: str) -> str:
        """Process a query asynchronously."""
        # Create task history and store query
        task_history = TaskHistory(user_query=query)
        self.history.append(task_history)

        # Analyze query and get tasks
        analyzer = QuestionAnalyzer(model=config.analyzer_model, callback=self.callback)
        result = await analyzer.analyze(query=query, deps=self.history)

        # Handle direct response or process tasks
        if isinstance(result, list):
            # Process each task
            for task_def in result:
                tool_manager = ToolManager(model=config.tool_manager_model, task_def=task_def, callback=self.callback)
                tools = await tool_manager.analyze()
                task = TaskRunner(
                    model=config.task_runner_model, task_def=task_def, toolset=tools, callback=self.callback
                )
                try:
                    task_result = await task.run(deps=self.history)
                except Exception as e:
                    print(f"Error running task {task_def.name}: {e}")
                    task_history.tasks.append(  # pylint: disable=no-member
                        TaskItem(query=task_def.name, result=f"Error running task {task_def.name}: {e}")
                    )
                    continue

                tool_results = extract_tool_results(task_result)
                task_history.tasks.append(  # pylint: disable=no-member
                    TaskItem(query=task_def.name, result=task_result.data, tool_results=tool_results)
                )
        # Get final result
        aggregator = Aggregator(model=config.aggregator_model, callback=self.callback)
        final_result = await aggregator.analyze(query=query, deps=self.history)

        # Update histories
        task_history.final_result = final_result
        return final_result

    @save_history()
    def ask_sync(self, query: str) -> str:
        """Process a query synchronously."""
        # Create task history and store query
        task_history = TaskHistory(user_query=query)
        self.history.append(task_history)

        # Analyze query and get tasks
        analyzer = QuestionAnalyzer(model=config.analyzer_model, callback=self.callback)
        result = analyzer.analyze_sync(query=query, deps=self.history)

        # Handle direct response or process tasks
        if result:
            # Process each task
            for cnt, task_def in enumerate(result):
                tool_manager = ToolManager(model=config.tool_manager_model, task_def=task_def, callback=self.callback)
                self._emit_status(task_def.name, "Analyzing task", 100 * (cnt + 1) // len(result))
                tools = tool_manager.analyze_sync()
                task = TaskRunner(
                    model=config.task_runner_model, task_def=task_def, toolset=tools, callback=self.callback
                )
                try:
                    self._emit_status(task_def.name, "Running task", None)
                    task_result = task.run_sync(deps=self.history)
                except Exception as e:
                    self._emit_status(task_def.name, f"Task failed: {str(e)}", None)
                    print(f"Error running task {task_def.name}: {e}")
                    task_history.tasks.append(  # pylint: disable=no-member
                        TaskItem(query=task_def.name, result=f"Error running task {task_def.name}: {e}")
                    )
                    continue

                # Extract tool results from task_result
                tool_results = extract_tool_results(task_result)
                task_history.tasks.append(  # pylint: disable=no-member
                    TaskItem(query=task_def.name, result=task_result.data, tool_results=tool_results)
                )

        # Get final result
        aggregator = Aggregator(model=config.aggregator_model, callback=self.callback)
        final_result = aggregator.analyze_sync(query=query, deps=self.history)

        # Update history
        task_history.final_result = final_result
        return final_result

    def _emit_status(self, task_name: str, status: str, progress: int | None):
        self.callback(TaskStatus(task_name=task_name, status=status, progress=progress))


if __name__ == "__main__":
    import asyncio

    # async def main():
    #     agentgenius = AgentGENius()
    #     query = "What's my operating system?"
    #     result = await agentgenius.ask(query)
    #     print(result)
    #     print(agentgenius.history)

    # asyncio.run(main())

    def status_callback(status: TaskStatus):
        print(f"Task: {status.task_name}", end="\t\t")
        print(f"Status: {status.status}", end="\t\t")
        if status.progress is not None:
            print(f"Progress: {status.progress}%", end="")
        print()

    def main():
        agentgenius = AgentGENius(callback=status_callback)
        query = "show RAM and free on my system"
        result = agentgenius.ask_sync(query)
        print(agentgenius.history)
        print(result)

    main()
