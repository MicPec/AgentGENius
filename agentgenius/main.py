from typing import Tuple

from dotenv import load_dotenv
from rich import print

from agentgenius.aggregator import Aggregator

# from agentgenius.builtin_tools import *
from agentgenius.history import History, TaskHistory, TaskItem
from agentgenius.task_managment import QuestionAnalyzer, TaskRunner
from agentgenius.tasks import TaskDef
from agentgenius.tools_management import ToolManager
from agentgenius.utils import save_history, save_task_history

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
        self.history = History(max_items=max_history)

    @save_history()
    async def ask(self, query: str) -> str:
        """Process a query asynchronously."""
        # Create task history and store query
        task_history = TaskHistory(user_query=query)
        self.history.append(task_history)

        # Analyze query and get tasks
        analyzer = QuestionAnalyzer(model=self.model)
        result = await analyzer.analyze(query, history=self.history)

        # Handle direct response or process tasks
        if isinstance(result, str):
            task_history.tasks.append(TaskItem(query="direct_response", result=result))
            final_result = result
        else:
            # Process each task
            for task_def in result:
                tool_manager = ToolManager(model=self.model, task_def=task_def)
                tools = await tool_manager.analyze()
                task = TaskRunner(model=self.model, task_def=task_def, toolset=tools)
                task_result = await task.run(deps=task_history)
                task_history.tasks.append(TaskItem(query=task_def.name, result=task_result.data))

            # Get final result
            aggregator = Aggregator(model=self.model)
            final_result = await aggregator.analyze(query, task_history)

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
        analyzer = QuestionAnalyzer(model=self.model)
        result = analyzer.analyze_sync(query=query, deps=self.history)

        # Handle direct response or process tasks
        if isinstance(result, str):
            task_history.tasks.append(TaskItem(query="direct_response", result=result))
            aggregator = Aggregator(model=self.model)
            final_result = aggregator.analyze_sync(query=result, deps=self.history)
        else:
            # Process each task
            for task_def in result:
                tool_manager = ToolManager(model=self.model, task_def=task_def)
                tools = tool_manager.analyze_sync()
                task = TaskRunner(model=self.model, task_def=task_def, toolset=tools)
                task_result = task.run_sync(deps=task_history)
                task_history.tasks.append(TaskItem(query=task_def.name, result=task_result.data))

            # Get final result
            aggregator = Aggregator(model=self.model)
            final_result = aggregator.analyze_sync(query=query, deps=self.history)

        # Update histories
        task_history.final_result = final_result
        return final_result


if __name__ == "__main__":
    agentgenius = AgentGENius()
    # print(agentgenius.ask_sync("what time is it?, what is the weather now?"))
    print(
        agentgenius.ask_sync(
            "która godzina? rot13 this: Rkcrpgvznk nytbevguz vf n cbchyne grpuavdhr hfrq va tnzr gurbel gb svaq gur bcgvzny zbir sbe n cynlre."
        )
    )
    print(agentgenius.history)
