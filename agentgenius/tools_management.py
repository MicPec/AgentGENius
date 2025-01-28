import tempfile
from pathlib import Path
from typing import Callable, Optional, TypeVar

from pydantic import BaseModel, Field

from agentgenius.agents import AgentDef, AgentParams
from agentgenius.builtin_tools import get_installed_packages
from agentgenius.tasks import Task, TaskDef
from agentgenius.tools import ToolSet
from agentgenius.utils import load_builtin_tools, load_generated_tools, search_frame

CACHE_DIR = Path(tempfile.gettempdir()) / "agentgenius" / "cache"
TOOLS_DIR = Path(tempfile.gettempdir()) / "agentgenius" / "tools"


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
        self.temp_dir = TOOLS_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.tool_request = tool_request
        self.task = Task(
            task_def=TaskDef(name="tool_request", query="Create a tool that will solve this task"),
            agent_def=AgentDef(
                model=model,
                name="tool manager",
                system_prompt=f"""Objective: As an expert Python developer, your task is to create a new tool function intended for use by an AI agent. Follow the requirements meticulously to ensure the function is robust, safe, and adheres to best practices.

Requirements:
1. Specific yet Generic:
- The function should address a specific task while maintaining generic applicability, allowing it to be reused in different contexts.
- Try to be as generic as possible, never hardcode any specific data, all custom values should be passed as arguments or keyword arguments.

2. Consider Scenarios and Edge Cases:
- Anticipate various use scenarios and edge cases that might arise when the function is executed.

3. Avoid Fake/Dummy Data:
- Ensure credibility by refraining from using any fake, dummy, example or placeholder data in your function.

4. Adhere to ToolRequest Specifications:
- Construct the function in line with the specified arguments (args) and keyword arguments (kwargs) given in the ToolRequest.

5. Type Hints:
- Incorporate Python type hints for all parameters and the return value, enhancing code readability and reliability.

6. Comprehensive Docstring:
- Write a short docstring that includes:
-- A description of the function’s purpose.
-- Documentation of the parameters.
-- A note of the return value(s).

7. Error Handling:
- Implement try/except blocks to handle potential errors gracefully, ensuring the function doesn’t fail unexpectedly.

8. PEP 8 Compliance:
- Follow the PEP 8 style guide to ensure your function adheres to Python coding standards.

9. Permitted Modules Only:
- Restrict yourself to using ONLY the modules available in the predefined environment:
-- Modules: {get_installed_packages()}
- Avoid using any third-party libraries and services that require API keys or credentials.

10. Self-contained Function:
- All necessary imports should reside within the function. Do not rely on external imports.
- As output, return ONLY generic types (dict, list, str, int, etc.)
- You can call functions from other functions 
- For big amounts of data, consider tool that save data to file and load data by other tool.
- Tools that save data (eg. dataframes, texts, images, etc.) should use `{CACHE_DIR}` folder for storing files.

11. User Safety:
- Craft the function with user safety as a priority. Under no circumstances should the function:
-- Delete files.
-- Expose sensitive information.
-- Execute or suggest any malicious code.
-- Engage in any illegal activities.

Example:
ToolRequest:
("tool_name": 'open_json_file', 'description': 'Open and read a JSON file', 'args': ('path',), 'kwargs': {{"mode": 'r'}})

ToolRequestResult:
def open_json_file(path: str, mode: str = 'r') -> dict:
    '''
    Open and read a JSON file from the given filepath.

    Args:
        path (str): The path to the JSON file to be opened.
        mode (str): The file mode for opening the file. Default is 'r' for read.

    Returns:
        dict: The contents of the JSON file as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    '''
    
    import json
    
    try:
        with open(path, mode) as f:
            return json.load(f)
    except FileNotFoundError as e:
        return f"Error: {{e}}"
    except json.JSONDecodeError as e:
        return f"Error: {{e}}"
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
            system_prompt=f"""Objective: You are an expert at selecting and creating tools necessary to accomplish a given task efficiently. Your goal is to identify and propose the optimal set of tools, both existing and potential, needed to solve the task at hand.

1. Instructions:
Understand the Task: Carefully review the task you are expected to solve. Break down the task into smaller components as needed to identify specific requirements.

2. Identify Tools:
List all existing tools that can be directly applied to parts of the task. Consider functions that are already available.
If the task involves dependencies, such as deriving one piece of information from another (e.g., obtaining a location from an IP address), ensure that tools for each step are included.
Add multiple tools for different parts of the task if needed.
Try to solve the task using existing tools whenever possible, it's better to combine existing tools rather than create new ones.
If the task doesn't require any tools (like explanation, etc.), return an empty ToolSet.

3. Tool Creation:
If you identify a gap where no existing tool meets the task requirements, propose a ToolRequest to create a new tool.
Ensure the proposed tool is simple, universal, and easy to reuse.
Name the tool using valid Python function naming conventions, ensuring it aptly describes the function for future identification and use.
Consider the necessary arguments (args) and keyword arguments (kwargs) for the tool, reflecting parameters like file paths, filenames, modes, etc.
Mind that all necessary data should be passed as arguments or keyword arguments.

4. Optimize Selection:
Prioritize using existing tools over creating new ones. Built-in tools are preferred over custom ones.
If multiple tools can serve the same purpose, select the one that is more commonly used or most efficient.

5. Information Sourcing:
For tasks requiring knowledge beyond readily available tools, consider using built-in 'web_search', 'scrape_webpage' or `extract_text_from_url` tools.
For data operations, consolidate all steps into one tool (eg. `read_and_analyze_data` that returns final result) or use files to pass data between tools. DO not pass large data (like dataframes) between tools.
Tools that save data (eg. dataframes, texts, images, etc.) should use `{CACHE_DIR}` folder for storing files. You should mention it in your tool request.

6. Deliverable:
Return a comprehensive list of applicable existing tools. Prefer built-in tools over custom ones.
Include a ToolRequest for any proposed new tools, clearly specifying the required functionality and parameters.
If all aspects of the task can be handled without additional tools, return an empty ToolSet.

7. Evaluation Criteria:
Efficiency and completeness of the proposed toolset in addressing the task.
Reusability and simplicity of any created tools.
Preference and utilization of existing and built-in solutions over novel tool creation.
""",
            params=AgentParams(
                result_type=ToolManagerResult,
                deps_type=TaskDef,
                retries=3,
            ),
        )
        self.task = Task(
            task_def=TaskDef(
                name="tool_manager",
                agent_def=self._agent,
                query=f"Select or create tools for this task: {str(task_def)}",
            )
        )

        @self.task._agent.system_prompt  # pylint: disable=protected-access
        async def get_available_tools():
            """Return a list of available tool names."""
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
            return f"Available tools:\n- Builtin tools: {', '.join(builtin_tools.keys())}\n- Generated tools: {', '.join(generated_tools.keys())}"

    async def analyze(self) -> ToolSet:
        result = await self.task.run()
        if isinstance(result.data, ToolManagerResult):
            if result.data.tool_request:
                requested_tools = ToolSet(
                    tools=[
                        await self._generate_tool(tool_request=tool_request)
                        for tool_request in result.data.tool_request
                    ]
                )
                return result.data.toolset | requested_tools
            return result.data.toolset
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
            try:
                tool = await tool_coder.get_tool()
            except Exception as e:
                return str(e)
            if isinstance(tool, Callable):
                return tool
            return ValueError("Failed to generate tool")
        raise ValueError("Invalid tool request")

    def _generate_tool_sync(self, tool_request: ToolRequest) -> Callable:
        if isinstance(tool_request, ToolRequest):
            tool_coder = ToolCoder(model=self.model, tool_request=tool_request)
            try:
                tool = tool_coder.get_tool_sync()
            except Exception as e:
                return str(e)
            if isinstance(tool, Callable):
                return tool
            return ValueError("Failed to generate tool")
        raise ValueError("Invalid tool request")
