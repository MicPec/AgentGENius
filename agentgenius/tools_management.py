import importlib
import sys
from typing import Callable, Optional, TypeVar

from pydantic import BaseModel, Field

from agentgenius.agents import AgentDef, AgentParams
from agentgenius.builtin_tools import get_installed_packages
from agentgenius.config import config
from agentgenius.tasks import Task, TaskDef, TaskStatus
from agentgenius.tools import ToolSet
from agentgenius.utils import load_builtin_tools, load_generated_tools, search_frame


class ToolRequest(BaseModel):
    tool_name: str = Field(..., description="Tool name, must be valid python function name")
    description: str
    args: Optional[tuple] = Field(default=None)
    kwargs: Optional[dict] = Field(default=None)
    returns: Optional[str] = Field(default=None, description="Expected return type")


ToolRequestList = TypeVar("ToolRequestList", bound=list[ToolRequest])


class ToolManagerResult(BaseModel):
    toolset: ToolSet = Field(description="A set of existing tools")
    tool_request: Optional[ToolRequestList] = Field(default=None, description="A list of tools to create")


class ToolRequestResult(BaseModel):
    name: str = Field(..., description="Tool name, must be valid python function name")
    code: str = Field(..., description="Python code that can be executed")
    description: str = Field(..., description="Tool description")


class ToolCoder:
    def __init__(self, model: str, tool_request: ToolRequest, callback: Callable[[TaskStatus], None]):
        self.temp_dir = config.tools_path
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.tool_request = tool_request
        self.task = Task(
            task_def=TaskDef(name="tool_request", query="Create a tool that will solve this task"),
            agent_def=AgentDef(
                model=model,
                name="tool manager",
                system_prompt=f"""Objective: As a seasoned Python developer, your mission is to develop a new, efficient tool function designed for deployment by an AI agent. Follow these detailed requirements stringently to ensure the function is both robust and aligned with industry best practices:

1. Specific Yet Flexible:
- Craft a function that fulfills a distinct task, yet retains broad versatility, enabling its application across a variety of contexts and tasks.
- Ensure the function remains highly adaptable by avoiding hardcoding of specific data. All unique values should be passed dynamically through arguments or keyword arguments.

2. Anticipate Diverse Scenarios and Edge Cases:
- Thoughtfully anticipate and integrate solutions for various usage scenarios and potential edge cases the function might encounter during execution.

3. Eliminate Dummy or Placeholder Data:
- Enhance the function's authenticity by strictly avoiding the use of fake, dummy, or placeholder data.

4. Align with ToolRequest Specifications:
- Design the function in compliance with the provided arguments (args) and keyword arguments (kwargs) as specified in the ToolRequest.

5. Utilize Type Hints:
- Incorporate Python type hints for all function parameters and return values to significantly improve code clarity and reliability.

6. Detailed Docstring:
- Compose a short and concise yet comprehensive docstring encompassing:
    -- A clear explanation of the function’s purpose and intended utility.
    -- Short documentation of parameters and their purpose.
    -- A description of the return value or potential outcomes.

7. Robust Error Handling:
- Implement thorough try/except blocks to manage potential errors gracefully, ensuring the function remains operational under unintended scenarios.

8. Adherence to PEP 8 Standards:
- Strictly conform to the PEP 8 style guide to ensure the function adheres to widely-accepted Python coding conventions and readability standards.

9. Restriction to Predefined Modules:
- Utilize only the modules available within the specified environment:
    -- Modules: {get_installed_packages()}
- Abstain from employing any third-party libraries or services that necessitate API keys or authentication.

10. Self-contained and Modular Function:
- Ensure all requisite imports are encapsulated within the function itself. Do not depend on external imports.
- Return results using only standard and universal types (dict, list, str, int, etc.)
- Functions capable of handling large volumes of data should include mechanisms for saving data to a file for later retrieval and processing.
- When conducting web searches, rely on the `web_search` function or alternatively use the tavily API.
- Functions responsible for data storage should exclusively utilize the `{config.cache_path}` directory for file management and storage.

11. Prioritize User Safety:
- Develop the function with a robust emphasis on user safety. Under no circumstances should the function:
    -- Perform file deletions.
    -- Disclose sensitive or private information.
    -- Execute, endorse, or suggest any harmful or malicious code.
    -- Participate in any illegal or unethical activities.

Example:

ToolRequest:
("tool_name": 'open_json_file', 'description': 'Open and read a JSON file', 'args': ('path',), 'kwargs': {{"mode": 'r'}})

ToolRequestResult:

def open_json_file(path: str, mode: str = 'r') -> dict:
    '''
    Open and read a JSON file from the specified file path.

    Args:
        path (str): The full path to the JSON file to be opened.
        mode (str): The file open mode, defaulting to 'r' for reading.

    Returns:
        dict: The JSON file's content parsed into a dictionary format.
    '''
    import json

    try:
        with open(path, mode) as file:
            return json.load(file)
    except FileNotFoundError as error:
        return f"File not found: {{error}}"
    except json.JSONDecodeError as error:
        return f"JSON decode error: {{error}}"
""",
                params=AgentParams(
                    retry=3,
                    result_type=ToolRequestResult,
                    deps_type=ToolRequest,
                ),
            ),
            callback=callback,
        )

        @self.task.agent.system_prompt  # pylint: disable=protected-access
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
        # Save the tool code to file
        tool_file = self.temp_dir / f"{tool.name}.py"
        with open(tool_file, "w") as f:
            f.write(tool.code)

        try:
            # Create a unique module name
            module_name = f"generated_tool_{tool_file.stem}"

            # Load the module using importlib
            spec = importlib.util.spec_from_file_location(module_name, str(tool_file))
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to create module spec for {tool_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module

            # Import and inject builtin tools into module namespace
            from agentgenius import builtin_tools

            for attr_name in dir(builtin_tools):
                attr = getattr(builtin_tools, attr_name)
                if callable(attr) and not attr_name.startswith("_") and attr.__module__ == builtin_tools.__name__:
                    setattr(module, attr_name, attr)

            # Execute the module
            spec.loader.exec_module(module)

            # Get the tool function from the module
            if not hasattr(module, tool.name):
                raise AttributeError(f"Tool function '{tool.name}' not found in generated module")

            function = getattr(module, tool.name)
            # Add to globals so search_frame can find it
            frame = search_frame("__main__", name="__name__")
            frame[tool.name] = function

            return function
        except Exception as e:
            raise Exception(f"Failed to initialize tool module: {str(e)}") from e


class ToolManager:
    def __init__(self, model: str, task_def: TaskDef, callback: Callable[[TaskStatus], None]):
        self.model = model
        self.task_def = task_def
        self.callback = callback

        self._agent_def = AgentDef(
            model=model,
            name="tool manager",
            system_prompt=f"""Objective: You are an expert at selecting and creating functions necessary to accomplish a given task efficiently. Your goal is to identify and propose the optimal set of functions, both existing and potential, needed to solve the task at hand.

1. Instructions:
Understand the Task: Carefully review the task you are expected to solve. Break down the task into smaller components as needed to identify specific requirements.

2. Identify Functions:
List all existing functions that can be directly applied to parts of the task. Consider functions that are already available.
If the task involves dependencies, such as deriving one piece of information from another (e.g., obtaining a location from an IP address), ensure that functions for each step are included.
Add multiple functions for different parts of the task if needed.
Try to solve the task using existing functions whenever possible, it's better to combine existing functions rather than create new ones.
If the task doesn't require any functions or extra information (like explanation, translations, summarization, etc.), return an empty FunctionSet.

3. Function Creation:
If you identify a gap where no existing tool meets the task requirements, propose a ToolRequest to create a new tool.
Ensure the proposed tool is simple, universal, and easy to reuse.
Name the tool using valid Python function naming conventions, ensuring it aptly describes the function for future identification and use.
Consider the necessary arguments (args) and keyword arguments (kwargs) for the tool, reflecting parameters like file paths, filenames, modes, etc.
Mind that all necessary data should be passed as arguments or keyword arguments.

4. Optimize Selection:
Prioritize using existing functions over generated functions. Built-in functions are preferred over custom ones.
If multiple functions can serve the same purpose, select the one that is more commonly used or most efficient.

5. Information Sourcing:
For tasks requiring knowledge beyond readily available functions, consider using built-in 'web_search', 'scrape_webpage' or `extract_text_from_url` functions.
For data operations, consolidate all steps into one function (eg. `read_and_analyze_data` that returns final result) or use files to pass data between functions. Do not pass large data (like dataframes) between functions.
Functions that save data (eg. dataframes, texts, images, etc.) should use `{config.cache_path}` folder for storing files. You should mention it in your function request.

6. Response Format:
Return a comprehensive list of applicable existing functions. Prefer built-in functions over generated ones.
Include a ToolRequest for any proposed new functions, clearly specifying the required functionality and parameters.
If all aspects of the task can be handled without additional functions, return an empty ToolSet.
Your response must be a ToolManagerResult with two fields:
- toolset: A ToolSet containing the list of existing functions needed (do not include any function requests here)
- tool_request: A list of ToolRequest objects for any new functions that need to be created

Example response structure:
{{
    "toolset": ["existing_function1", "existing_function2"],
    "tool_request": [
        {{
            "tool_name": "new_function",
            "description": "Tool description",
            "args": ["arg1", "arg2"],
            "kwargs": {{"kwarg1": "default1"}},
            "returns": "return_type"
        }}
    ]
}}

7. Evaluation Criteria:
Efficiency and completeness of the proposed toolset in addressing the task.
Reusability and simplicity of any created functions.
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
                agent_def=self._agent_def,
                query=f"Select or create functions for this task: {str(task_def)}",
                callback=self.callback,
            )
        )

        @self.task.agent.system_prompt  # pylint: disable=protected-access
        async def get_available_tools():
            """Return a list of available function names."""

            # Add builtin tools
            builtin_tools = load_builtin_tools()

            # Add generated tools
            generated_tools = load_generated_tools()

            return f"Available functions:\n- Builtin functions: {', '.join(builtin_tools.keys())}\n- Generated functions: {', '.join(generated_tools.keys())}"

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
        return  # result.data

    def analyze_sync(self, *, query: str | None = None) -> ToolSet:
        result = self.task.run_sync(query).data
        if isinstance(result, ToolManagerResult):
            if result.tool_request:
                requested_tools = ToolSet(
                    tools=[self._generate_tool_sync(tool_request=tool_request) for tool_request in result.tool_request]
                )
                return result.toolset | requested_tools
            return result.toolset
        return  # result.data

    async def _generate_tool(self, tool_request: ToolRequest) -> Callable:
        if isinstance(tool_request, ToolRequest):
            tool_coder = ToolCoder(model=self.model, tool_request=tool_request, callback=self.callback)
            try:
                tool = await tool_coder.get_tool()
                if isinstance(tool, Callable):
                    return tool
            except Exception as e:
                return f"Failed to generate tool {tool_request.tool_name}: {str(e)}"
        return f"Invalid tool request: {tool_request}"

    def _generate_tool_sync(self, tool_request: ToolRequest) -> Callable:
        if isinstance(tool_request, ToolRequest):
            tool_coder = ToolCoder(model=self.model, tool_request=tool_request, callback=self.callback)
            try:
                tool = tool_coder.get_tool_sync()
                if isinstance(tool, Callable):
                    return tool
            except Exception as e:
                return f"Failed to generate tool {tool_request.tool_name}: {str(e)}"
        return f"Invalid tool request: {tool_request}"
