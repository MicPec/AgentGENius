import asyncio

from agentgenius.agents import BaseAgent
from pydantic_ai import RunContext, Tool
from dotenv import load_dotenv
import nest_asyncio
from agentgenius.tools import ToolSet
from datetime import date

nest_asyncio.apply()
load_dotenv()


def ask_user_tool(ctx: RunContext[str], question: str) -> str:
    """Ask the user a question"""
    return input(question + " ")


tools = ToolSet(ask_user_tool)


agent = BaseAgent(model="openai:gpt-4o-mini", system_prompt="You are a helpful assistant.", toolset=tools)


@agent.system_prompt
def add_the_date() -> str:
    return f"The date is {date.today()}."


def main():
    result = agent.run_sync("is today my birthday?")
    print(result.data)


if __name__ == "__main__":
    main()
