from types import NoneType
from typing import Tuple

from dotenv import load_dotenv
from rich import print

from agentgenius.aggregator import Aggregator

# from agentgenius.builtin_tools import *
from agentgenius.history import History, TaskHistory, TaskItem
from agentgenius.task_managment import QuestionAnalyzer, TaskRunner
from agentgenius.tasks import TaskDef
from agentgenius.tools_management import ToolManager
from agentgenius.utils import save_history

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
        result = await analyzer.analyze(query=query, deps=self.history)

        # Handle direct response or process tasks
        if result is not None:
            # Process each task
            for task_def in result:
                tool_manager = ToolManager(model=self.model, task_def=task_def)
                tools = await tool_manager.analyze()
                task = TaskRunner(model=self.model, task_def=task_def, toolset=tools)
                task_result = await task.run(deps=task_history)

                # Extract tool results from task_result
                tool_results = []
                for msg in task_result._all_messages:
                    if msg.kind == "response" and hasattr(msg, "parts"):
                        for part in msg.parts:
                            if hasattr(part, "tool_name"):
                                # Find the corresponding tool return
                                tool_return = next(
                                    (
                                        ret.parts[0].content
                                        for ret in task_result._all_messages
                                        if ret.kind == "request"
                                        and hasattr(ret, "parts")
                                        and hasattr(ret.parts[0], "tool_call_id")
                                        and ret.parts[0].tool_call_id == part.tool_call_id
                                    ),
                                    None,
                                )
                                if tool_return:
                                    tool_results.append(
                                        {
                                            "tool": part.tool_name,
                                            "args": str(part.args.args_json)
                                            if hasattr(part.args, "args_json")
                                            else str(part.args.args_dict)
                                            if hasattr(part.args, "args_dict")
                                            else None,
                                            "result": str(tool_return) if tool_return is not None else "",
                                        }
                                    )

                task_history.tasks.append(  # pylint: disable=no-member
                    TaskItem(query=task_def.name, result=task_result.data, tool_results=tool_results)
                )

        # Get final result
        aggregator = Aggregator(model=self.model)
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
        analyzer = QuestionAnalyzer(model=self.model)
        result = analyzer.analyze_sync(query=query, deps=self.history)

        # Handle direct response or process tasks
        if result:
            # Process each task
            for task_def in result:
                tool_manager = ToolManager(model=self.model, task_def=task_def)
                tools = tool_manager.analyze_sync()
                task = TaskRunner(model=self.model, task_def=task_def, toolset=tools)
                task_result = task.run_sync(deps=task_history)

                # Extract tool results from task_result
                tool_results = self._extract_tool_results(task_result)
                task_history.tasks.append(  # pylint: disable=no-member
                    TaskItem(query=task_def.name, result=task_result.data, tool_results=tool_results)
                )

        # Get final result
        aggregator = Aggregator(model=self.model)
        final_result = aggregator.analyze_sync(query=query, deps=self.history)

        # Update history
        task_history.final_result = final_result
        return final_result

    def _extract_tool_results(self, task_result):
        # Extract tool results from task_result
        tool_results = []
        for msg in task_result._all_messages:
            if msg.kind == "response" and hasattr(msg, "parts"):
                for part in msg.parts:
                    if hasattr(part, "tool_name"):
                        # Find the corresponding tool return
                        tool_return = next(
                            (
                                ret.parts[0].content
                                for ret in task_result._all_messages
                                if ret.kind == "request"
                                and hasattr(ret, "parts")
                                and hasattr(ret.parts[0], "tool_call_id")
                                and ret.parts[0].tool_call_id == part.tool_call_id
                            ),
                            None,
                        )
                        if tool_return:
                            tool_results.append(
                                {
                                    "tool": part.tool_name,
                                    "args": str(part.args.args_json)
                                    if hasattr(part.args, "args_json")
                                    else str(part.args.args_dict)
                                    if hasattr(part.args, "args_dict")
                                    else None,
                                    "result": str(tool_return) if tool_return is not None else None,
                                }
                            )
        return tool_results


if __name__ == "__main__":
    import asyncio

    async def main():
        agentgenius = AgentGENius()
        query = "What's my operating system?"
        result = await agentgenius.ask(query)
        print(result)
        print(agentgenius.history)

    asyncio.run(main())
