import tempfile
from pathlib import Path
from typing import Callable, TypeVar, Union, Any, Optional, Tuple

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry
from rich import print

from agentgenius.agents import AgentDef, AgentParams
from agentgenius.builtin_tools import *
from agentgenius.builtin_tools import _get_builtin_tools, get_installed_packages
from agentgenius.tasks import Task, TaskDef
from agentgenius.tools import ToolDef, ToolSet

logfire.configure(send_to_logfire="if-token-present")

load_dotenv()


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
        self.items.append(item)
        if len(self) > self.max_items:
            self.items.pop(0)


class QuestionAnalyzer:
    def __init__(self, query: str):
        self.query = query
        self._agent = AgentDef(
            model="openai:gpt-4o",
            name="task analyzer",
            system_prompt="""You are an expert at breaking down complex tasks into smaller, manageable pieces.
            Think step by step, what are the steps to solve this task and what information are needed to do it?
            Focus on creating clear, detailed, effective, and actionable subtasks that can be executed independently.
            In the field 'query', put the command for an AI agent, not question

            Example:
            Question: What movies are playing today in my local cinema?
            Steps:
            1. Find the user location and current time.
            2. Get the cinema location in the user's area.
            3. Get the movie schedule for the local cinema.

            Question: Search for file in my home directory
            Steps:
            1. Identify the user's operating system.
            2. Get the user's name and home directory.
            3. Use the user's operating system to search for the file.
            """,
            params=AgentParams(
                result_type=list[TaskDef],
            ),
        )

    async def analyze(self) -> list[TaskDef]:
        result = await Task(task_def=TaskDef(name="task_analysis", agent_def=self._agent, query=self.query)).run()
        return sorted(result.data, key=lambda x: x.priority)

    def analyze_sync(self) -> list[TaskDef]:
        result = Task(task_def=TaskDef(name="task_analysis", agent_def=self._agent, query=self.query)).run_sync()
        return sorted(result.data, key=lambda x: x.priority)


class ToolRequest(BaseModel):
    tool_name: str = Field(..., description="Tool name, must be valid python function name")
    description: str
    args: Optional[tuple] = Field(default=None)
    kwargs: Optional[dict] = Field(default=None)


class ToolRequestResult(BaseModel):
    name: str
    code: str


class ToolCoder:
    def __init__(self, tool_request: ToolRequest, sender: Task | None = None):
        self.temp_dir = Path(tempfile.gettempdir()) / "agentgenius_tools"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.tool_request = tool_request
        self.task = Task(
            task_def=TaskDef(name="tool_request", query="Create a tool: "),
            agent_def=AgentDef(
                model="openai:gpt-4o",
                name="tool manager",
                system_prompt=f"""You are an expert python developer.
You are asked to create a new tool function that will be used by an AI agent.
Requirements:
1. Function should be focused and solve one specific task
2. The function will be called with the arguments and kwargs specified in the ToolRequest
3. Include type hints for parameters and return value
4. Write a clear docstring with description, args, and returns
5. Handle errors gracefully with try/except
6. Follow PEP 8 style guide
7. Use only these modules: {get_installed_packages()}
8. The function should be self contained (all imports inside the function)
9. Make sure the code is safe for user. NEVER delete any files or show secret information or execute any malicious code. Don't do anything illegal.

Example:
ToolRequest: ("tool_name": 'open_json_file', 'description': 'Open and read a JSON file', 'args': ('path',), 'kwargs': {{"mode": 'r'}})
ToolRequestResult:
name: open_json_file
code:
def open_json_file(path, mode='r'):
    import json
    # open JSON file in given path with given mode
    with open(path, mode) as f:
        return json.load(f)
""",
                params=AgentParams(
                    retry=3,
                    result_type=ToolRequestResult,
                    deps_type=ToolRequest,
                ),
            ),
        )
        # self.sender = sender if sender else self.task

    async def get_tool(self) -> str:
        tool_coder = await self.task.run(self.tool_request)
        try:
            function = self.save_tool(tool_coder.data)
            return function
        except Exception as e:
            return str(e)

    def get_tool_sync(self) -> str:
        tool_coder = self.task.run_sync(self.tool_request)
        try:
            function = self.save_tool(tool_coder.data)
            return function
        except Exception as e:
            return str(e)

    def save_tool(self, tool: ToolRequestResult) -> Union[Callable, str]:
        with open(self.temp_dir / f"{tool.name}.py", "w") as f:
            f.write(tool.code)
        try:
            exec(tool.code, globals())
            return globals()[tool.name]
        except Exception as e:
            raise Exception(f"Failed to add tool to module: {str(e)}")


def tool_request(tool_request: ToolRequest):
    """Create a new tool to solve the task"""
    tool_coder = ToolCoder(tool_request)
    # Use sync version since this is called from sync context
    return tool_coder.get_tool_sync()


class ToolManager:
    def __init__(self, task_def: TaskDef):
        self.task_def = task_def

        self._agent = AgentDef(
            model="openai:gpt-4o",
            name="tool manager",
            system_prompt=f"""You are an expert at selecting and creating  tools for a given task. 
            Think what tools are needed to solve this task and propose them.
            Return ToolSet if you have all needed tools ready,
            or ToolRequest if you need to create a new tool.
            Created tools must be universal and easy to reuse later.
            Available tools are: {self.get_available_tools()}
            """,
            # """,system_prompt=f"""You are an expert at creating and selecting tools for a given task.ALWAYS GENERATE A TOOL.
            # Think what tools are needed to solve this task and propose them.
            # If no appropriate tool is found, create a new tool by calling 'tool_request' function and then add the requested tool to the ToolSet.
            # Be proactive - always think about possible tools to generate.
            # You MUST use the 'tool_request' before adding the tool to the ToolSet.
            # Created tools must be easy to reuse later.
            # Available tools are: {{self.get_available_tools()}}.
            # Example:
            # Question: List all files in my home directory?
            # Toolset: ["get_home_directory", "list_directory"]
            # Example:
            # Question: Open pdf file
            # call 'tool_request' with:
            #     ToolRequest: ("tool_name": 'open_pdf_file', 'description': 'Open and read a PDF file', 'args': ('path',), 'kwargs': {{"mode": 'r'}})
            # return:
            # Toolset: ["open_pdf_file"] + you can add more existing tools
            # schema: {ToolRequest.model_json_schema()}
            # """,
            params=AgentParams(
                result_type=Union[ToolSet, list[ToolRequest]],
                deps_type=TaskDef,
                retries=2,
            ),
            # tools=ToolSet([tool_request]),
        )

        self.task = Task(
            task_def=TaskDef(
                name="tool_manager",
                agent_def=self._agent,
                # toolset=ToolSet([tool_request]),
                query=f"Select or create tools for this task: {str(task_def)}",
            )
        )

    def get_available_tools(self):
        """Return a list of available tool names. DO NOT CALL THESE TOOLS.
        Just pass them to the other agents and let them to use them."""
        toolset = ToolSet(_get_builtin_tools())
        return toolset.all()

    async def analyze(self) -> ToolSet:
        result = await self.task.run()
        if isinstance(result.data, dict) and "tools" in result.data:
            return ToolSet(tools=result.data["tools"])
        return result.data

    def analyze_sync(self, query: str | None = None) -> ToolSet:
        result = self.task.run_sync(query)
        if isinstance(result.data, ToolSet):
            return result.data
        if isinstance(result.data, list):
            return ToolSet(tools=[self._generate_tool(tool_request=tool_request) for tool_request in result.data])
        if isinstance(result.data, ToolRequest):
            return ToolSet(tools=[self._generate_tool(tool_request=result.data)])
        raise ValueError("Failed to analyze tools")

    def _generate_tool(self, tool_request: ToolRequest) -> Callable:
        if isinstance(tool_request, ToolRequest):
            tool_coder = ToolCoder(tool_request)
            tool = tool_coder.get_tool_sync()
            if isinstance(tool, Callable):
                return tool
            raise ValueError("Failed to generate tool")
        raise ValueError("Invalid tool request")


class Aggregator:
    def __init__(self, history: TaskHistory):
        self.history = history
        self.task_def = TaskDef(
            name="aggregator",
            query="Analyze full history of tasks and results and generate a final answer. Answer in language of the user first query.",
            priority=10,
            agent_def=AgentDef(
                model="openai:gpt-4o",
                name="aggregator",
                system_prompt="You are an expert at synthesizing information and providing clear, direct answers.",
                params=AgentParams(
                    result_type=str,
                    deps_type=TaskHistory,
                ),
            ),
        )

        self.task = Task(task_def=self.task_def)

    async def analyze(self):
        return (await self.task.run(self.history)).data

    def analyze_sync(self):
        print(self.history)
        return self.task.run_sync(self.history).data


class AgentGENius:
    """
    Main agent class that orchestrates the process of analyzing queries, managing tasks,
    and aggregating results. Supports both synchronous and asynchronous operations.
    """

    def __init__(
        self, model: str = "openai:gpt-4o", system_prompt: str = "You are a helpful assistant.", max_history: int = 10
    ):
        """
        Initialize the AgentGENius.

        Args:
            model: The model identifier to use for the agent
            system_prompt: The system prompt for the agent
            max_history: Maximum number of items to keep in task history
        """
        self.task_def = TaskDef(
            name="agent",
            query="",
            priority=10,
            agent_def=AgentDef(
                model=model,
                name="agent",
                system_prompt=system_prompt,
            ),
        )
        self.history = TaskHistory(max_items=max_history)

    # Main public interface methods
    async def ask(self, query: str) -> str:
        """
        Process a query asynchronously and return the result.

        Args:
            query: The query to process

        Returns:
            str: The final aggregated response
        """
        self._store_query(query)
        tasks, task_history = await self._analyze_query(query)
        await self._process_all_tasks(tasks, task_history)
        result = await self._get_final_result(task_history)
        self._update_history(tasks, result)
        return result

    def ask_sync(self, query: str) -> str:
        """
        Synchronous version of ask method.

        Args:
            query: The query to process

        Returns:
            str: The final aggregated response
        """
        self._store_query(query)
        tasks, task_history = self._analyze_query_sync(query)
        self._process_all_tasks_sync(tasks, task_history)
        result = self._get_final_result_sync(task_history)
        self._update_history(tasks, result)
        return result

    # Query analysis methods
    async def _analyze_query(self, query: str) -> Tuple[list[TaskDef], TaskHistory]:
        """Analyze the query and break it down into tasks asynchronously."""
        analyzer = QuestionAnalyzer(query=query)
        tasks = await analyzer.analyze()
        return tasks, self._create_task_history(analyzer.query)

    def _analyze_query_sync(self, query: str) -> Tuple[list[TaskDef], TaskHistory]:
        """Synchronous version of query analysis."""
        analyzer = QuestionAnalyzer(query=query)
        tasks = analyzer.analyze_sync()
        return tasks, self._create_task_history(analyzer.query)

    # Task processing methods
    async def _process_all_tasks(self, tasks: list[TaskDef], task_history: TaskHistory) -> None:
        """Process all tasks asynchronously."""
        for task_def in tasks:
            await self._process_single_task(task_def, task_history)

    def _process_all_tasks_sync(self, tasks: list[TaskDef], task_history: TaskHistory) -> None:
        """Process all tasks synchronously."""
        for task_def in tasks:
            self._process_single_task_sync(task_def, task_history)

    async def _process_single_task(self, task_def: TaskDef, task_history: TaskHistory) -> None:
        """Process a single task asynchronously."""
        tool_manager = ToolManager(task_def=task_def)
        tools = await tool_manager.analyze()
        task = Task(task_def=task_def, agent_def=self.task_def.agent_def, toolset=tools)
        result = await task.run(deps=task_history)
        task_history.append({"task": task_def.name, "result": result.data})

    def _process_single_task_sync(self, task_def: TaskDef, task_history: TaskHistory) -> None:
        """Process a single task synchronously."""
        tool_manager = ToolManager(task_def=task_def)
        tools = tool_manager.analyze_sync()
        task = Task(task_def=task_def, agent_def=self.task_def.agent_def, toolset=tools)
        result = task.run_sync(deps=task_history)
        task_history.append({"task": task_def.name, "result": result.data})

    # Result aggregation methods
    async def _get_final_result(self, task_history: TaskHistory) -> str:
        """Aggregate results from all tasks asynchronously."""
        aggregator = Aggregator(history=task_history)
        return await aggregator.analyze()

    def _get_final_result_sync(self, task_history: TaskHistory) -> str:
        """Aggregate results from all tasks synchronously."""
        aggregator = Aggregator(history=task_history)
        return aggregator.analyze_sync()

    # History management methods
    def _store_query(self, query: str) -> None:
        """Store the initial query in history."""
        self.history.append({"user_query": query})

    def _create_task_history(self, query: str) -> TaskHistory:
        """Create a new task history for the current query."""
        task_history = TaskHistory()
        task_history.append({"user_query": query})
        return task_history

    def _update_history(self, tasks: list[TaskDef], result: str) -> None:
        """Update the main history with the final result."""
        if tasks:
            self.history[-1]["result"] = result


if __name__ == "__main__":
    agentgenius = AgentGENius()
    print(agentgenius.ask_sync("jaki dziś mamy dzień i moje IP?"))
    print(agentgenius.history)

    # tr = ToolRequest(
    #     tool_name="open_mp3_file",
    #     description="Open and play a mp3 file",
    #     args=("path",),
    # )
    # print(tool_request(tr))
    # open_mp3_file = tool_request(tr)
    # print(open_mp3_file("/home/michal/Music/Agi skladanka/1 When I'm Gone.mp3"))
