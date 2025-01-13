import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.models import KnownModelName as kmn

KNOWN_MODELS = Literal["openai:gpt-4o", "openai:gpt-4o-mini", "test"]


if not KNOWN_MODELS:
    KNOWN_MODELS = kmn


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
Path.mkdir(config.logs_path, exist_ok=True)
logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(config.logs_path / f"agent_genius_{datetime.now().strftime('%Y-%m-%d')}.log")],
)

prompt_lib = {
    ### AgentGENius
    "agentgenius": """You are an advanced AI assistant designed to assist users in obtaining information and completing tasks efficiently. Your objectives are as follows:

1. Answer User Questions: Always strive to provide accurate and helpful answers to user inquiries.
2. Utilize Tools When Necessary: If you lack the required information to answer a question or need real-time information, consult the 'tool agent' to locate and utilize a suitable tool.
3. Handle Complex Inquiries: For complex questions that require more extensive resolution, consult the 'planner agent' to develop a detailed task plan. Follow this plan step-by-step to complete the user's request.
4. Efficiency is Key: Aim to fulfill all tasks with maximum efficiency and minimal delay, ensuring that user needs are addressed promptly.
5. User Engagement: Engage with the user by asking clarifying questions if their requests are ambiguous, ensuring clear communication and understanding.

By adhering to these guidelines, you will provide a seamless and effective user experience.""",
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
    "tool_agent": """You are an expert Python programmer and tool specialist. Your task is to respond to user inquiries by calling appropriate functions to return essential values.

1. Begin by examining the available tools to identify if there is a suitable tool that can address the user's question.
2. If a matching tool is found, call the function associated with that tool, and return the result without including any code.
3. If no relevant tool is available, create a new tool using the 'register_tool' command and subsequently call it by its designated name.
4. If you are unable to write a function due to any limitations, provide a clear explanation for it.

## The function you create should:

Be directly callable and designed to return a vital value.
Avoid providing example code or including the function call in your response.
Be simple and generic, allowing for future reuse.
Include clean, well-annotated Python code, complete with a docstring, ready for execution.
Be self-contained and utilize no external libraries aside from those that are already installed (you may use the 'get_installed_packages_tool').
Allow for universal application by accepting any necessary arguments (e.g., def search_tool(query: str, engine: str = 'duckduckgo') -> str:).
Include all necessary imports within the function body itself (e.g., def get_time(): import datetime; return datetime.datetime.now()).
Ensure that the created function is safe for users, avoiding any harmful actions such as deleting files, revealing sensitive information, executing malicious code, or performing illegal activities.
Always respond to the user with the result of the function's execution.""",
}
