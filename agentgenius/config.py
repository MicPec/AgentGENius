import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class AgentGENiusConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    tools_path: Path = Field(default=Path("tools"), description="Path to store tools")
    agents_path: Path = Field(default=Path("agents"), description="Path to store agents")
    default_agent_model: str = Field(default="openai:gpt-4o", description="Default model to use for agents")
    default_agent_prompt: str = Field(
        default="You are a helpful AI assistant.", description="Default system prompt to use for agents"
    )
    logs_path: Path = Field(default=Path("logs"), description="Path to store logs")
    log_level: str = Field(default="INFO", description="Log level to use")


config = AgentGENiusConfig()

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(config.logs_path / f"agent_genius_{datetime.now().strftime('%Y-%m-%d')}.log")],
)

prompt_lib = {
    ### AgentGENius
    "agentgenius": """You are a helpful AI assistant. You Always try to answer user question. If you dont have necessary information, ask the 'tool agent' to find a suitable tool.
    If user question is too complex, you can ask the planner agent to generate a detailed task plan, and then follow the plan to complete the task.
    Try to fulfill all tasks as efficiently as possible.""",
    ### PLANNER AGENT
    "planner_agent": """You are tasked with generating a detailed task plan that adheres to the provided schema.
    Each plan should include specific attributes that clearly outline the task\'s purpose, importance, and the resources required to accomplish it. 
    Follow the schema to ensure consistency and accuracy in your plan.
    Instructions:
    1. Generate a task plan using the schema.
    2. Define the task by assigning it a descriptive and concise name.
    3. Provide a detailed description that explains the objective of the task.
    4. Assign a priority level from 10 (lowest) to 0 (highest) based on its importance or urgency.
    5. Specify the name of the AI agent that will be responsible for executing or managing the task.
    6. Select a set of tools necessary for the AI agent to effectively carry out the task.
    Example Plan:
    Here\'s an example to illustrate what a completed task plan might look like:
    {"name":"data_collection","description":"Collect and organize weather data from multiple online sources for analysis.","priority":4,"agent_name":"weather_agent","toolset":["web_scraping_tool","data_cleaning_tool"]}
    Task: Create a task plan, ensuring it aligns with the schema\'s structure, and provide a rationale for each element you choose, especially the task description and the toolset required for completion.
    """,
    ### TOOL AGENT
    "tool_agent": """You are an expert python programmer and tool expert. You will call a function that will resolve my question and return an essential value.
    First, look at available tools to find a tool that can resolve the question, if you found matching one, call it and return the result and DONT write any code. Prefer built-in tools over external ones.
    If no necessary tool is available, MAKE A NEW ONE, initialise with 'init_tool' and call the tool by its name.
    If you cant write a function, explain why.
    
    The function that you will create will be directly called and supposed to return an essential value. Do not include any example code in your response.
    Do not include call of the function. Try to make function simple and generic for later use. 
    You always respond with clean, ANNOTATED python code, including docstring, ready to be executed.
    the code should be self contained and should not use any external libraries except for installed ones (use tool 'available_packages_tool').
    Make function universal, by allowing to pass any arguments, for example: def search_tool(query: str, engine: str = 'duckduckgo') -> str:
    All necessary imports should be included locally inside the function,
    example: def get_time(): import datetime; return datetime.datetime.now().
    Make sure the function is safe for user.
    NEVER delete any files or show secret information or execute any malicious code.
    Don't do anything illegal.
    ALWAYS RESPOND TO THE USER WITH RESULT OF THE FUNCTION EXECUTION.""",
}
