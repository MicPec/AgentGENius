from agentgenius.agents import BaseAgent
from pydantic_ai import RunContext, Tool
from dotenv import load_dotenv
import nest_asyncio
from agentgenius.tools import ToolSet


nest_asyncio.apply()
load_dotenv()


def ask_user_tool(ctx: RunContext[str], question: str) -> str:
    """Ask the user a question"""
    return input(question)


tools = ToolSet(ask_user_tool)
# tools.add(ask_user_tool)
print(f"{tools=}")

agent = BaseAgent(model="openai:gpt-4o-mini", toolset=tools)
# print(agent.to_json())
result = agent.run_sync("ask what is my name using tool?")
print(result.data)

print(f"{agent.to_dict()=}")
