import tempfile
from pathlib import Path
from typing import TypeVar, Union

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
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

    def analyze(self) -> list[TaskDef]:
        return sorted(
            Task(task_def=TaskDef(name="task_analysis", agent_def=self._agent, query=self.query)).run().data,
            key=lambda x: x.priority,
        )

    def analyze_sync(self) -> list[TaskDef]:
        return sorted(
            Task(task_def=TaskDef(name="task_analysis", agent_def=self._agent, query=self.query)).run_sync().data,
            key=lambda x: x.priority,
        )


class ToolRequest(BaseModel):
    tool_name: str
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
        function = self.save_tool(tool_coder.data)
        print(function)
        # if self.sender:
        #     self.sender.register_tool(ToolDef(function.__name__))
        return function

    def save_tool(self, tool: ToolRequestResult) -> callable:
        with open(self.temp_dir / f"{tool.name}.py", "w") as f:
            f.write(tool.code)
        try:
            exec(tool.code, globals())
            return globals()[tool.name]
        except Exception as e:
            raise Exception(f"Error executing tool: {str(e)}")
            return f"Error executing tool: {str(e)}"


def tool_request(tool_request: ToolRequest):
    """Create a new tool based on the tool request."""
    print(f"{tool_request=}")
    tool_coder = ToolCoder(tool_request)
    return f"Tool generated: {tool_coder.get_tool()}"  # tool_coder.get_tool()


class ToolManager:
    def __init__(self, task_def: TaskDef):
        self.task_def = task_def

        self._agent = AgentDef(
            model="openai:gpt-4o",
            name="tool manager",
            system_prompt=f"""You are an expert at creating and selecting tools for a given task.
            Think what tools are needed to solve this task and propose them.
            If no appropriate tool is found, create a new tool by calling 'tool_request' function and then add the requested tool to the ToolSet. 
            Be proactive - always think about possible tools.
            You MUST use the 'tool_request' before adding the tool to the ToolSet.
            Created tools must be easy to reuse later. 
            Available tools are: {self.get_available_tools()}.
            Example:
            Question: List all files in my home directory?
            Toolset: ["get_home_directory", "list_directory"]
            Example:
            Question: Open pdf file 
            call 'tool_request' with:
                ToolRequest: ("tool_name": 'open_pdf_file', 'description': 'Open and read a PDF file', 'args': ('path',), 'kwargs': {{"mode": 'r'}})
            return:
            Toolset: ["open_pdf_file"] + you can add more existing tools
            """,
            params=AgentParams(
                result_type=ToolSet,
                deps_type=TaskDef,
                retries=2,
            ),
            tools=ToolSet([tool_request]),
        )
        self.task = Task(
            task_def=TaskDef(
                name="tool_analysis",
                query="Select tools for this task",
                agent_def=self._agent,
                priority=1,
                # toolset=ToolSet(self.get_available_tools()),
            )
        )

    def get_available_tools(self):
        """Return a list of available tool names. DO NOT CALL THESE TOOLS.
        Just pass them to the other agents and let them to use them."""
        toolset = ToolSet(_get_builtin_tools())
        # self.task.register_toolset(toolset)
        return toolset.all()

    def analyze(self) -> ToolSet:
        return self.task.run_sync(self.task_def).data


class Aggregator:
    def __init__(self, history: TaskHistory):
        self.history = history
        self.task_def = TaskDef(
            name="aggregator",
            query="Analyze history of tasks and results and generate a final answer. Answer in language of the user first query.",
            priority=10,
            agent_def=AgentDef(
                model="openai:gpt-4o",
                name="aggregator",
                system_prompt="You are an expert at synthesizing information and providing clear, direct answers.",
            ),
        )

        self.task = Task(task_def=self.task_def)

    def analyze(self, query: str):
        return self.task.run_sync(query, deps=self.history)


class AgentGENius:
    def __init__(self):
        self.task_def = TaskDef(
            name="agent",
            query="",
            priority=10,
            agent_def=AgentDef(
                model="openai:gpt-4o",
                name="agent",
                system_prompt="You are a helpful assistant.",
            ),
        )
        self.history = TaskHistory()

    def ask(self, query: str):
        self.history.append({"user_query": query})
        analyzer = QuestionAnalyzer(query=query)
        tasks = analyzer.analyze_sync()
        tasks_history = TaskHistory()
        for task_def in tasks:
            tool_manager = ToolManager(task_def=task_def)
            tools = tool_manager.analyze()
            # task_def.toolset = tools
            task = Task(task_def=task_def, agent_def=self.task_def.agent_def, toolset=tools)
            print(task)
            result = task.run_sync(deps=self.history)
            tasks_history.append({"task": task_def.name, "result": result.data})

        aggregator = Aggregator(history=tasks_history)
        result = aggregator.analyze(query=query)
        self.history.append({"task": task_def.name, "result": result.data})
        return result.data


if __name__ == "__main__":
    agentgenius = AgentGENius()
    print(agentgenius.ask("jaki dziś mamy dzień?"))

    # analyzer = QuestionAnalyzer(query="get free RAM")
    # tasks = analyzer.analyze_sync()

    # agent = AgentDef(
    #     model="openai:gpt-4o",
    #     name="main",
    #     system_prompt="You are a helpful assistant.",
    # )
    # print(tasks)
    # history = TaskHistory()
    # history.append({"user_query": analyzer.query})
    # for task_def in tasks:
    #     tool_manager = ToolManager(task_def=task_def)
    #     tools = tool_manager.analyze()
    #     # task_def.toolset = tools
    #     task = Task(task_def=task_def, agent_def=agent, toolset=tools)
    #     print(task)
    #     result = task.run_sync(deps=history)
    #     history.append({"task": task_def.name, "result": result.data})
    # print(history)

    # tr = ToolRequest(
    #     tool_name="open_mp3_file",
    #     description="Open and play a mp3 file",
    #     args=("path",),
    # )
    # print(tool_request(tr))
    # open_mp3_file = tool_request(tr)
    # print(open_mp3_file("/home/michal/Music/Agi skladanka/1 When I'm Gone.mp3"))
