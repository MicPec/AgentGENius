import tempfile
from pathlib import Path
from typing import Callable, Optional, TypeVar

from pydantic import BaseModel, Field

from agentgenius.agents import AgentDef, AgentParams
from agentgenius.builtin_tools import get_installed_packages
from agentgenius.tasks import Task, TaskDef
from agentgenius.tools import ToolSet
from agentgenius.utils import load_builtin_tools, load_generated_tools, search_frame


class ToolRequest(BaseModel):
    tool_name: str = Field(..., description="Tool name, must be valid python function name")
    description: str
    args: Optional[tuple] = Field(default=None)
    kwargs: Optional[dict] = Field(default=None)


ToolRequestList = TypeVar("ToolRequestList", bound=list[ToolRequest])


class ToolRequestResult(BaseModel):
    name: str = Field(..., description="Tool name, must be valid python function name")
    code: str = Field(..., description="Python code that can be executed")


class ToolCoder:
    def __init__(self, model: str, tool_request: ToolRequest):
        self.temp_dir = Path(tempfile.gettempdir()) / "agentgenius_tools"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.tool_request = tool_request
        self.task = Task(
            task_def=TaskDef(name="tool_request", query="Create a tool: "),
            agent_def=AgentDef(
                model=model,
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
7. Use ONLY these modules: {get_installed_packages()}
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

    def save_tool(self, tool: ToolRequestResult) -> Callable:
        with open(self.temp_dir / f"{tool.name}.py", "w") as f:
            f.write(tool.code)
        try:
            frame = search_frame("__main__", name="__name__")
            exec(tool.code, frame)
            return frame[tool.name]
        except Exception as e:
            raise Exception(f"Failed to add tool to module: {str(e)}") from e


class ToolManagerResult(BaseModel):
    toolset: ToolSet = Field(default_factory=ToolSet, description="A set of existing tools")
    tool_request: Optional[ToolRequestList] = Field(default=None, description="A list of tools to create")


class ToolManager:
    def __init__(self, model: str, task_def: TaskDef):
        self.model = model
        self.task_def = task_def

        self._agent = AgentDef(
            model=model,
            name="tool manager",
            system_prompt=f"""You are an expert at selecting and creating  tools for a given task. 
            Available tools are: {self.get_available_tools()}
            Think what tools are needed to solve this task and propose them.
            Return ToolSet of existing tools that are applicable to the task,
            and ToolRequest if you need to create a new tools.
            Created tools must be simple, universal and easy to reuse later. 
            Prefer existing tools over creating new ones.
            If no tools are needed, return an empty ToolSet.
            """,
            params=AgentParams(
                result_type=ToolManagerResult,
                deps_type=TaskDef,
                retries=2,
            ),
        )

        self.task = Task(
            task_def=TaskDef(
                name="tool_manager",
                agent_def=self._agent,
                query=f"Select or create tools for this task: {str(task_def)}",
            )
        )

    def get_available_tools(self):
        """Return a list of available tool names. DO NOT CALL THESE TOOLS.
        Just pass them to the other agents and let them to use them."""
        tools = ToolSet()

        # Add builtin tools
        builtin_tools = load_builtin_tools()
        if builtin_tools:
            tools.add(builtin_tools)

        # Add generated tools
        generated_tools = load_generated_tools()
        if generated_tools:
            tools.add(generated_tools)

        # print(f"{tools=}", tools)
        return tools.all()

    async def analyze(self) -> ToolSet:
        result = await self.task.run()
        if isinstance(result.data, dict) and "tools" in result.data:
            return ToolSet(tools=result.data["tools"])
        return result.data

    def analyze_sync(self, *, query: str | None = None) -> ToolSet:
        result = self.task.run_sync(query)
        if isinstance(result.data, ToolManagerResult):
            if result.data.tool_request:
                requested_tools = ToolSet(
                    tools=[
                        self._generate_tool_sync(tool_request=tool_request) for tool_request in result.data.tool_request
                    ]
                )
                return result.data.toolset | requested_tools
            return result.data.toolset
        return result.data

    async def _generate_tool(self, tool_request: ToolRequest) -> Callable:
        if isinstance(tool_request, ToolRequest):
            tool_coder = ToolCoder(model=self.model, tool_request=tool_request)
            tool = await tool_coder.get_tool()
            if isinstance(tool, Callable):
                return tool
            raise ValueError("Failed to generate tool")
        raise ValueError("Invalid tool request")

    def _generate_tool_sync(self, tool_request: ToolRequest) -> Callable:
        if isinstance(tool_request, ToolRequest):
            tool_coder = ToolCoder(model=self.model, tool_request=tool_request)
            tool = tool_coder.get_tool_sync()
            if isinstance(tool, Callable):
                return tool
            raise ValueError("Failed to generate tool")
        raise ValueError("Invalid tool request")
